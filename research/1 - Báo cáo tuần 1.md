Em chào thầy, nhóm cảm ơn thầy đã nhận lời hướng dẫn cuộc thi này, theo nhóm có đọc qua thể lệ của cuộc thi thì có thấy Track 4 là phù hợp nhất, bởi vì nó có cả nhiệm vụ AI lẫn tối ưu nhúng trên thiết bị NVIDA Jetson
Tuần qua, nhóm đã tìm hiểu thử code của các đội đạt giải năm trước thì có nhận thấy điểm chung là các đội đều thực hiện tăng cường dữ liệu bằng các dataset bên ngoài 
Tuy nhiên, ở các đội, nhóm vẫn còn thấy chưa chú trọng vào việc tối ưu trên thiết bị Jetson
Qua thảo luận thì nhóm có ý tưởng như sau: 
- Trước tiên, thực hiện train thử trên model nhỏ, do yêu cầu của cuộc thi là FPS trong khoảng 10 đến 25 FPS, nên nhóm muốn thực hiện thử các bước sau (chuyển sang TensorRT) để xem FPS như thế nào, từ đó thực hiện cải thiện rồi chuyển sang có model phức tạp hơn 
- Khó khăn:
	- Tuy nhiên, nhóm không có thiết bị để thực hiện thử, tính đến hiện tại, nhóm đã train được model yolov11n, nhưng độ chính xác không cao, do mất cân bằng ở bộ dữ liệu (Jetson AGX Orin)
	- Thời gian train model khá lâu, các tài nguyên miễn phí như google colab hay kaggle không đủ để thực hiện train (nhóm chỉ thực hiện train thử yolov11n, nhưng mất gần 8 giờ để hoàn thành)

