#include "face_detector.h"

#include "vnn_face.h"
#include "vnn_kit.h"

#include "util.h"
NS_GPUPIXEL_BEGIN

FaceDetector::FaceDetector() {
  // 初始化，设置VNN的日志级别为全部
  VNN_SetLogLevel(VNN_LOG_LEVEL_ALL);
#if defined(GPUPIXEL_IOS) || defined(GPUPIXEL_ANDROID)
  // 获取移动端人脸检测模型的路径
  auto model_path = Util::getResourcePath("face_mobile[1.0.0].vnnmodel");
#elif defined(GPUPIXEL_WIN) || defined(GPUPIXEL_MAC) || defined(GPUPIXEL_LINUX)
  // 获取PC端人脸检测模型的路径
  auto model_path = Util::getResourcePath("face_pc[1.0.0].vnnmodel");
#endif
  // 模型参数
  const void* argv[] = {
      model_path.c_str(),
  };

  const int argc = sizeof(argv) / sizeof(argv[0]);
  // 创建人脸检测器
  VNN_Result ret = VNN_Create_Face(&vnn_handle_, argc, argv);
}

FaceDetector::~FaceDetector() {
  // 销毁人脸检测器
  if (vnn_handle_ > 0) {
    VNN_Destroy_Face(&vnn_handle_);
  }
}

/**
 * @brief 注册人脸检测回调函数。
 *
 * @param callback  人脸检测回调函数。
 * @return int  0 表示成功。
 */
int FaceDetector::RegCallback(FaceDetectorCallback callback) {
  _face_detector_callbacks.push_back(callback);
  return 0;
}

/**
 * @brief 注册人脸框检测回调函数。
 *
 * @param callback  人脸框检测回调函数。
 * @return int  0 表示成功。
 */
int FaceDetector::RegFaceBoundingBoxCallback(FaceBoundingBoxCallback callback) {
  _face_bounding_box_callbacks.push_back(callback);
  return 0;
}

/**
 * @brief 获取人脸框信息。
 *
 * @return std::vector<float>
 * 包含所有检测到的人脸框坐标的vector，每个人脸框由四个浮点数表示：x0(左),
 * y0(上), x1(右), y1(下)。
 */
std::vector<float> FaceDetector::GetFaceBoundingBoxes() {
  return _face_bounding_boxes;
}

/**
 * @brief 设置当前选中的人脸索引。
 *
 * @param index 要选中的人脸索引。
 */
void FaceDetector::SetSelectedFaceIndex(int index) {
  if (index >= 0 && index < _detected_faces_count) {
    _selected_face_index = index;
  }
}

/**
 * @brief 获取当前选中的人脸索引。
 *
 * @return int 当前选中的人脸索引。
 */
int FaceDetector::GetSelectedFaceIndex() const {
  return _selected_face_index;
}

/**
 * @brief 获取检测到的人脸数量。
 *
 * @return int 检测到的人脸数量。
 */
int FaceDetector::GetDetectedFacesCount() const {
  return _detected_faces_count;
}

/**
 * @brief 检测图像中的人脸。
 *
 * @param data  图像数据指针。
 * @param width  图像宽度。
 * @param height  图像高度。
 * @param fmt  图像模式格式。
 * @param type  图像帧类型。
 * @return int  0 表示成功，-1 表示失败。
 */
int FaceDetector::Detect(const uint8_t* data,
                         int width,
                         int height,
                         GPUPIXEL_MODE_FMT fmt,
                         GPUPIXEL_FRAME_TYPE type) {
  // 检查人脸检测器句柄是否有效。如果句柄为0，表示人脸检测器未初始化或已销毁。
  // 如果句柄无效，则函数返回-1，表示检测失败。
  if (vnn_handle_ == 0) {
    // 打印错误信息，方便调试
    // LOGE("人脸检测器未初始化或已销毁");
    return -1;  // 返回错误码
  }

  // 设置是否使用278个关键点。`use_278pts`
  // 是一个成员变量，控制是否使用更精细的关键点检测。
  VNN_Set_Face_Attr(vnn_handle_, "_use_278pts", &use_278pts);

  // 构建输入图像结构体，用于传递给人脸检测库。
  VNN_Image input;
  input.width = width;    // 图像宽度
  input.height = height;  // 图像高度
  input.channels = 4;     // 图像通道数，这里假设是 RGBA
  // 根据图像类型设置像素格式。`type` 参数来自外部，指示图像的颜色格式。
  switch (type) {
    case GPUPIXEL_FRAME_TYPE_RGBA8888: {
      // 如果是 RGBA8888，则设置为 BGRA8888，因为 VNN
      // 库可能使用 BGRA 顺序。
      input.pix_fmt = VNN_PIX_FMT_BGRA8888;

    } break;
    case GPUPIXEL_FRAME_TYPE_YUVI420: {
      // 如果是 YUVI420，则设置相应的像素格式。
      input.pix_fmt = VNN_PIX_FMT_YUVI420;
    } break;
    default:
      break;  // 默认情况，不做处理。
  }

  input.data = (VNNVoidPtr)data;  // 图像数据指针
  // 根据模式格式设置模式格式。`fmt` 参数指示图像是用于视频还是图片。
  if (fmt == GPUPIXEL_MODE_FMT_VIDEO) {
    input.mode_fmt = VNN_MODE_FMT_VIDEO;  // 如果是视频模式，则设置模式为视频。
  }

  if (fmt == GPUPIXEL_MODE_FMT_PICTURE) {
    // 如果是图片模式，则设置模式为图片。
    input.mode_fmt = VNN_MODE_FMT_PICTURE;
  }

  // 图像方向格式，使用默认值。
  input.ori_fmt = VNN_ORIENT_FMT_DEFAULT;

  // 定义人脸检测输出结构体，用于接收人脸检测结果。
  VNN_FaceFrameDataArr faceArr, detectionArr;
  memset(&faceArr, 0x00, sizeof(VNN_FaceFrameDataArr));
  memset(&detectionArr, 0x00, sizeof(VNN_FaceFrameDataArr));
  // 使用CPU进行人脸检测。`VNN_Apply_Face_CPU` 是 VNN 库提供的函数，用于在 CPU
  // 上进行人脸检测。
  VNN_Result ret = VNN_Apply_Face_CPU(vnn_handle_, &input, &faceArr);

  // 注意：在 VNN_Apply_Face_CPU 之后调用
  ret = VNN_Get_Face_Attr(vnn_handle_, "_detection_data", &detectionArr);

  // 清空之前的人脸框数据
  _face_bounding_boxes.clear();

  // 处理人脸框数据
  if (detectionArr.facesNum > 0) {
    // 更新检测到的人脸数量
    _detected_faces_count = detectionArr.facesNum;

    // 遍历所有检测到的人脸
    for (int i = 0; i < detectionArr.facesNum; i++) {
      // 获取人脸框坐标
      _face_bounding_boxes.push_back(
          detectionArr.facesArr[i].faceRect.x0);  // 左
      _face_bounding_boxes.push_back(
          detectionArr.facesArr[i].faceRect.y0);  // 上
      _face_bounding_boxes.push_back(
          detectionArr.facesArr[i].faceRect.x1);  // 右
      _face_bounding_boxes.push_back(
          detectionArr.facesArr[i].faceRect.y1);  // 下
    }

    // 调用人脸框回调函数
    for (auto cb : _face_bounding_box_callbacks) {
      cb(_face_bounding_boxes);
    }
  } else {
    // 如果没有检测到人脸，将人脸数量设为0
    _detected_faces_count = 0;
    _selected_face_index = 0;
  }

  // 如果检测到人脸，处理所有人脸的关键点
  if (faceArr.facesNum > 0) {
    // 遍历所有检测到的人脸
    for (int face_idx = 0; face_idx < faceArr.facesNum; face_idx++) {
      // 存储当前人脸的关键点坐标
      std::vector<float> landmarks;

      // 遍历当前人脸的所有关键点
      for (int i = 0; i < faceArr.facesArr[face_idx].faceLandmarksNum; i++) {
        // 将关键点坐标添加到向量中
        landmarks.push_back(faceArr.facesArr[face_idx].faceLandmarks[i].x);  // 添加 x 坐标
        landmarks.push_back(faceArr.facesArr[face_idx].faceLandmarks[i].y);  // 添加 y 坐标
      }

      // 计算并添加额外的关键点（106）
      auto point_x = (faceArr.facesArr[face_idx].faceLandmarks[102].x + faceArr.facesArr[face_idx].faceLandmarks[98].x) / 2;
      auto point_y = (faceArr.facesArr[face_idx].faceLandmarks[102].y + faceArr.facesArr[face_idx].faceLandmarks[98].y) / 2;
      landmarks.push_back(point_x);
      landmarks.push_back(point_y);

      // 计算并添加额外的关键点（107）
      point_x = (faceArr.facesArr[face_idx].faceLandmarks[35].x + faceArr.facesArr[face_idx].faceLandmarks[65].x) / 2;
      point_y = (faceArr.facesArr[face_idx].faceLandmarks[35].y + faceArr.facesArr[face_idx].faceLandmarks[65].y) / 2;
      landmarks.push_back(point_x);
      landmarks.push_back(point_y);

      // 计算并添加额外的关键点（108）
      point_x = (faceArr.facesArr[face_idx].faceLandmarks[70].x + faceArr.facesArr[face_idx].faceLandmarks[40].x) / 2;
      point_y = (faceArr.facesArr[face_idx].faceLandmarks[70].y + faceArr.facesArr[face_idx].faceLandmarks[40].y) / 2;
      landmarks.push_back(point_x);
      landmarks.push_back(point_y);

      // 计算并添加额外的关键点（109）
      point_x = (faceArr.facesArr[face_idx].faceLandmarks[5].x + faceArr.facesArr[face_idx].faceLandmarks[80].x) / 2;
      point_y = (faceArr.facesArr[face_idx].faceLandmarks[5].y + faceArr.facesArr[face_idx].faceLandmarks[80].y) / 2;
      landmarks.push_back(point_x);
      landmarks.push_back(point_y);

      // 计算并添加额外的关键点（110）
      point_x = (faceArr.facesArr[face_idx].faceLandmarks[81].x + faceArr.facesArr[face_idx].faceLandmarks[27].x) / 2;
      point_y = (faceArr.facesArr[face_idx].faceLandmarks[81].y + faceArr.facesArr[face_idx].faceLandmarks[27].y) / 2;
      landmarks.push_back(point_x);
      landmarks.push_back(point_y);
      
      // 调用回调函数，传递当前人脸的关键点、索引和总人脸数
      for (auto cb : _face_detector_callbacks) {
        cb(landmarks, face_idx, faceArr.facesNum);
      }
    }
  } else {
    std::vector<float> landmarks;
    // 调用回调函数，传递当前人脸的关键点、索引和总人脸数
    for (auto cb : _face_detector_callbacks) {
      cb(landmarks, 0, faceArr.facesNum);
    }
  }

  return 0;  // 返回成功
}

NS_GPUPIXEL_END
