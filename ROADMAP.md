# Intelligent RoadEye — Roadmap

## Current Version (June 2026)

- [x] VGG16 fine-tuned on merged road defect dataset
- [x] Dataset balancing — augmentation + random sampling
- [x] GPU-accelerated training (CUDA, mixed precision)
- [x] Sliding window inference with confidence thresholding
- [x] Real-time Flask dashboard
- [x] Bounding box visualization per patch
- [x] Detection summary, severity meter, trend charts
- [x] Image and video upload support
- [x] Export report as JSON

---

## Planned Improvements

### Model
- [ ] Train on larger dataset (5,000+ images per class)
- [ ] Add a road region detector to filter out non-road patches before classification
- [ ] Experiment with ResNet50 and EfficientNetB0 for accuracy comparison
- [ ] Add crack severity classification — hairline, medium, severe

### Inference
- [ ] Non-Maximum Suppression (NMS) to reduce overlapping boxes
- [ ] ONNX export for faster CPU inference
- [ ] TensorRT optimization for edge deployment on Jetson Nano

### Dashboard
- [ ] GPS coordinate tagging for detected defects
- [ ] Heatmap overlay showing defect density across the road
- [ ] Export detection report as PDF with annotated image
- [ ] Session history — compare multiple uploads

### Deployment
- [ ] Deploy on AWS EC2 with public URL
- [ ] Edge deployment on NVIDIA Jetson Nano for drone integration
- [ ] REST API endpoint for integration with other systems
