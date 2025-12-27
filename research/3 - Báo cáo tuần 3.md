Em chào thầy, tuần này theo dự kiến là nhóm sẽ nộp được trên trang đánh giá của cuộc thi, tuy nhiên, BTC cuộc thi vẫn chưa cho phép nhóm nộp thử kết quả inference. Tuần vừa qua, nhóm đã train được trên các mô hình: RTDETRX, SODETR với kết quả như sau: 
- RTDETR X (của Ultralytics): Sau 47 epochs được F1 Score là 0.66
- SO DETR: Là một mô hình được cải thiện dựa trên RT DETR L dành riêng cho small object , tuy nhiên, thời gian train của model này khá lâu, nhóm chỉ train được 10 epochs và thu được F1 Score là 0.587, so với RTDETR X là 0.624 khi được train 9 epoch. 
- Tuy nhiên, thời gian infer của RTDETR X có phần lớn hơn so với SO DETR, nên tuần tiếp theo nhóm sẽ train trên cả hai model để so sánh 
Về nhiệm vụ infer thử trên edge device mà thầy cung cấp, nhóm đã chuyển model rtdetr x sang onnx, fps nhận được là 0.77 FPS, rất thấp so với 10 FPS yêu cầu của BTC, tuần này, nhóm sẽ tiếp tục train và tối ưu model dựa trên kết quả eval trên web của BTC
