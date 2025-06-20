# trt_inference.py
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import cv2

class TRTInference:
    """TensorRT推理引擎"""
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # 加载引擎
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.input_shape = self.engine.get_binding_shape(0)
        
        # 分配输入/输出内存
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
    
    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream
    
    def preprocess(self, image):
        """预处理输入图像"""
        # 应用CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(image)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = cv2.resize(img, (256, 256))
        
        # 转换为Tensor
        tensor = img.astype(np.float32).reshape(1, 1, 256, 256)
        return tensor
    
    def infer(self, image_path):
        """执行推理"""
        # 读取并预处理图像
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        input_data = self.preprocess(img)
        
        # 复制数据到设备
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 执行推理
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # 复制结果回主机
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        cuda.memcpy_dtoh_async(self.outputs[1]['host'], self.outputs[1]['device'], self.stream)
        self.stream.synchronize()
        
        # 获取输出
        seg_output = self.outputs[0]['host'].reshape(1, 256, 256)
        cls_output = self.outputs[1]['host'].reshape(1, 2)
        
        return seg_output, cls_output
    
    def visualize_results(self, image_path, seg_output, cls_output):
        """可视化结果"""
        img = cv2.imread(image_path)
        img = cv2.resize(img, (256, 256))
        
        # 处理分割结果
        seg_mask = (seg_output > 0.5).astype(np.uint8) * 255
        seg_mask = cv2.resize(seg_mask, (img.shape[1], img.shape[0]))
        
        # 处理分类结果
        benign_prob = cls_output[0][0]
        malignant_prob = cls_output[0][1]
        diagnosis = "良性" if benign_prob > malignant_prob else "恶性"
        confidence = max(benign_prob, malignant_prob)
        
        # 绘制结果
        overlay = cv2.addWeighted(img, 0.7, cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
        cv2.putText(overlay, f"诊断: {diagnosis} ({confidence:.2f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return overlay

# 使用示例
if __name__ == "__main__":
    # 初始化推理引擎
    trt_engine = TRTInference("thyroid_model.trt")
    
    # 执行推理
    image_path = "test_image.png"
    seg_mask, cls_logits = trt_engine.infer(image_path)
    
    # 可视化结果
    result_img = trt_engine.visualize_results(image_path, seg_mask, cls_logits)
    cv2.imwrite("result.png", result_img)
