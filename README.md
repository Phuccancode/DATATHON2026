# DATATHON 2026 - Round 1

Repository cho cuộc thi DATATHON 2026 - Round 1, bao gồm 3 phần chính: MCQs, Exploratory Data Analysis (EDA), và Forecasting Model.

## Cấu trúc thư mục

```
DATATHON/
├── data/                      # Dữ liệu cuộc thi
│   ├── products.csv           # Thông tin sản phẩm
│   ├── customers.csv          # Thông tin khách hàng
│   ├── promotions.csv         # Thông tin khuyến mãi
│   ├── geography.csv          # Thông tin địa lý
│   ├── orders.csv             # Đơn hàng
│   ├── order_items.csv        # Chi tiết đơn hàng
│   ├── payments.csv           # Thanh toán
│   ├── shipments.csv         # Vận chuyển
│   ├── returns.csv            # Trả hàng
│   ├── reviews.csv            # Đánh giá
│   ├── inventory.csv          # Kho hàng
│   ├── web_traffic.csv        # Lưu lượng truy cập web
│   ├── sales.csv              # Doanh số hàng ngày
│   ├── sample_submission.csv  # File mẫu nộp bài
│   └── baseline.ipynb         # Notebook baseline
│
├── part1/                     # Phần 1: MCQs
│   └── part1_mcqs.ipynb       # Notebook trả lời 10 câu hỏi MCQ
│
├── part2/                     # Phần 2: EDA
│   └── part2.ipynb            # Notebook phân tích dữ liệu khám phá
│
├── part3/                     # Phần 3: Forecasting Model
│   └── part3_forecast_model.ipynb  # Notebook dự báo Revenue & COGS
│
├── .venv/                     # Virtual environment Python
├── pyproject.toml             # Cấu hình project
└── README.md                  # File này
```

## Thiết lập môi trường

### 1. Tạo virtual environment (nếu chưa có)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc
.venv\Scripts\activate     # Windows
```

### 2. Cài đặt dependencies

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn lightgbm catboost optuna statsmodels shap tqdm openpyxl
```

**Các thư viện chính:**
- `pandas`, `numpy`: Xử lý và phân tích dữ liệu
- `scikit-learn`: Chia tập dữ liệu, đánh giá mô hình, tiền xử lý
- `matplotlib`, `seaborn`, `plotly`: Trực quan hóa dữ liệu
- `statsmodels`: Phân tích chuỗi thời gian (ADF, ACF/PACF)
- `lightgbm`, `catboost`: Mô hình gradient boosting
- `optuna`: Tối ưu hóa siêu tham số
- `shap`: Giải thích mô hình
- `tqdm`: Hiển thị tiến độ

## Mục tiêu bài toán

Xây dựng hệ thống dự báo doanh thu (Revenue) và chi phí hàng bán (COGS) từ dữ liệu lịch sử, bao gồm các giai đoạn:

1. **Tiền xử lý dữ liệu**: Đồng bộ định dạng, làm sạch dữ liệu, xử lý giá trị thiếu
2. **Phân tích khám phá (EDA)**: Hiểu cấu trúc dữ liệu, phát hiện pattern và xu hướng
3. **Feature Engineering**: Tạo đặc trưng từ dữ liệu thô để phục vụ mô hình
4. **Huấn luyện mô hình**: Thử nghiệm nhiều thuật toán và tối ưu hóa
5. **Ensemble**: Kết hợp nhiều mô hình để tăng độ ổn định và chính xác

## Quy trình thực hiện

### 1. Khám phá và tiền xử lý dữ liệu

Trong notebook `part2/part2.ipynb`:

- Đọc và kiểm tra các bảng dữ liệu (sales, traffic, promotions, sample_submission, ...)
- Xác định phạm vi thời gian: Train (04/2012–12/2022), Test (01/2023–06/2024)
- Đồng bộ định dạng cột: chuyển `Date` sang datetime, ép kiểu dữ liệu phù hợp
- Xử lý giá trị thiếu và loại bỏ outliers trong dữ liệu bán hàng
- Tổng hợp dữ liệu thành "Single Source of Truth" theo ngày: tính tổng Revenue và COGS
- Điền giá trị thiếu cho traffic, ghép dữ liệu khuyến mãi để xác định thời gian khuyến mãi

### 2. Phân tích dữ liệu và trực quan hóa

Trong notebook `part2/part2.ipynb`:

- **Phân tích chuỗi thời gian**:
  - Phân rã thành phần (decompose) để xác định xu hướng và mùa vụ
  - Kiểm định ADF để kiểm tra tính dừng của chuỗi
  - Vẽ ACF/PACF để hiểu độ tự tương quan

- **Phân tích các yếu tố ảnh hưởng**:
  - Hiệu quả doanh thu theo sản phẩm/danh mục
  - Tác động của khuyến mãi
  - Hành vi khách hàng theo nhóm/quốc gia
  - Hiệu quả vận hành kho

- **Trực quan hóa và Insight**:
  - Xác định chu kỳ mùa vụ (sales tăng/giảm theo tháng/quý)
  - Ảnh hưởng của ngày lễ/ngày trả lương/khuyến mãi
  - Mối quan hệ giữa traffic và doanh thu

- **Feature Engineering**: Tạo hơn 60 biến mới từ 7 nhóm:
  - Lag features (doanh thu các ngày trước đó)
  - Rolling statistics (trung bình và thống kê trượt)
  - Đặc trưng lịch (tháng, ngày trong tuần, tuần trong năm, ngày lễ, ngày trả lương)
  - Fourier cho thành phần mùa vụ phi tuyến
  - Xu hướng chung và tín hiệu ngoại sinh (traffic, khuyến mãi)

### 3. Huấn luyện mô hình

Trong notebook `part3/part3_forecast_model.ipynb`:

- **Chia dữ liệu theo thời gian**:
  - Training: đến hết năm 2021
  - Validation: năm 2022
  - Test: dự báo năm 2023-2024

- **Mô hình baseline**:
  - Naive Forecast (doanh thu ngày mai = ngày trước)
  - Seasonal Naive (dùng giá trị cùng ngày tuần trước)
  - Trung bình trượt

- **Huấn luyện mô hình chính**:
  - LightGBM, XGBoost, CatBoost
  - Seed averaging: chạy với nhiều random_state khác nhau và lấy trung bình
  - Tối ưu siêu tham số với Optuna (~50 trials)
  - Early stopping để tránh overfitting

- **Ensemble**: Kết hợp có trọng số các dự báo (ví dụ: 40% LightGBM, 20% XGBoost, 40% CatBoost)

- **Đánh giá**: Sử dụng MAE, RMSE, R² trên tập Validation

### 4. Dự báo và xuất kết quả

- **Dự báo đa bước (multi-step)**:
  - Dự báo tuần tự 548 ngày (năm 2023-2024)
  - Phương pháp recursive forecasting: mỗi bước dùng kết quả bước trước

- **Tính toán COGS**:
  - Sử dụng tỷ lệ biên lợi nhuận trung bình (COGS/Revenue) theo từng tháng
  - Áp dụng cho giá trị dự báo
  - Đảm bảo ràng buộc kinh doanh: COGS < Revenue

- **Kết quả đầu ra**: File `submission.csv` với các cột `Date`, `Revenue`, `COGS`

## Tổng kết quy trình

| Giai đoạn | Nội dung chính |
|-----------|----------------|
| EDA & Kiểm định | Phân tích chuỗi thời gian, xác nhận xu hướng tăng và mùa vụ, kiểm định ADF |
| Feature Engineering | Tạo 60+ biến từ 7 nhóm: lag, rolling, lịch, sự kiện, Fourier, xu hướng, ngoại sinh |
| Baseline | Thiết lập benchmark từ Naive, Seasonal, Rolling models |
| Huấn luyện & Tinh chỉnh | LightGBM, XGBoost, CatBoost + Optuna + Seed averaging + Ensemble |
| Dự báo & Xuất file | Dự báo tuần tự, áp dụng ràng buộc Revenue & COGS, sinh submission.csv |

## Hướng dẫn chạy lại kết quả

### Part 1: MCQs (Multiple Choice Questions)

Mở notebook `part1/part1_mcqs.ipynb` và chạy tất cả các cells.

**Kết quả:** 10 câu trả lời MCQ với các câu hỏi về:
- Khoảng thời gian giữa các đơn hàng
- Biên lợi nhuận theo phân khúc sản phẩm
- Lý do trả hàng phổ biến nhất
- Tỷ lệ bounce rate theo nguồn traffic
- Tỷ lệ đơn hàng có khuyến mãi
- Số đơn hàng trung bình theo nhóm tuổi
- Doanh thu theo khu vực
- Phương thức thanh toán phổ biến nhất cho đơn hàng bị hủy
- Tỷ lệ trả hàng theo kích thước
- Giá trị thanh toán trung bình theo số lần trả góp

**Cách chạy:**
```bash
jupyter notebook part1/part1_mcqs.ipynb
```

### Part 2: Exploratory Data Analysis (EDA)

Mở notebook `part2/part2.ipynb` và chạy các cells theo trình tự.

**Nội dung phân tích:**
- Khám phá dữ liệu khách hàng, sản phẩm, đơn hàng
- Phân tích hành vi mua sắm
- Phân tích đánh giá và trả hàng
- Phân tích lưu lượng web
- Phân tích theo thời gian

**Cách chạy:**
```bash
jupyter notebook part2/part2.ipynb
```

### Part 3: Forecasting Model

Mở notebook `part3/part3_forecast_model.ipynb` và chạy theo trình tự.

**Mục tiêu:** Dự báo `Revenue` và `COGS` hàng ngày cho 548 ngày (01/2023–06/2024).

**Cách chạy:**
```bash
jupyter notebook part3/part3_forecast_model.ipynb
```

**Quy trình:**
1. Tạo đặc trưng từ dữ liệu lịch sử
2. Huấn luyện mô hình LightGBM, XGBoost, CatBoost
3. Tối ưu siêu tham số với Optuna
4. Ensemble các mô hình
5. Dự báo tuần tự cho tập test
6. Tính toán COGS từ Revenue dự báo

**Kết quả đầu ra:**
- File `submission.csv` với định dạng:
  ```
  Date,Revenue,COGS
  2012-07-04,5123547.94,3982991.19
  ...
  ```

## Dữ liệu

### Master Data (Dữ liệu tham chiếu)
- `products.csv`: Thông tin sản phẩm (ID, tên, giá, COGS, phân khúc, danh mục, kích thước)
- `customers.csv`: Thông tin khách hàng (ID, độ tuổi, giới tính, vị trí)
- `promotions.csv`: Thông tin khuyến mãi
- `geography.csv`: Thông tin địa lý theo mã ZIP

### Transaction Data (Dữ liệu giao dịch)
- `orders.csv`: Đơn hàng (ID, khách hàng, ngày, trạng thái, địa chỉ)
- `order_items.csv`: Chi tiết đơn hàng (sản phẩm, số lượng, giá, khuyến mãi)
- `payments.csv`: Thanh toán (phương thức, số lần trả góp, giá trị)
- `shipments.csv`: Vận chuyển (đơn vị, thời gian, phí)
- `returns.csv`: Trả hàng (lý do, số lượng)
- `reviews.csv`: Đánh giá (sao, bình luận)

### Operational Data (Dữ liệu vận hành)
- `inventory.csv`: Kho hàng (sản phẩm, số lượng, vị trí)
- `web_traffic.csv`: Lưu lượng web (nguồn, lượt xem, bounce rate)

### Analytical Data (Dữ liệu phân tích)
- `sales.csv`: Doanh số hàng ngày (Date, Revenue, COGS)
- `sample_submission.csv`: File mẫu nộp bài

## Kết quả MCQs (Part 1)

| Câu | Đáp án | Giải thích |
|-----|--------|------------|
| Q1  | C      | Khoảng thời gian trung bình giữa các đơn hàng: 144 ngày |
| Q2  | D      | Phân khúc có biên lợi nhuận cao nhất: Standard (~31.3%) |
| Q3  | B      | Lý do trả hàng phổ biến nhất cho Streetwear: wrong_size (7,626 lần) |
| Q4  | C      | Nguồn traffic có bounce rate thấp nhất: email_campaign (~0.45%) |
| Q5  | C      | Tỷ lệ đơn hàng có khuyến mãi: 38.66% |
| Q6  | A      | Nhóm tuổi có số đơn hàng trung bình cao nhất: 55+ (~5.4 đơn/khách) |
| Q7  | C      | Khu vực có doanh thu cao nhất: East (~7.6 tỷ) |
| Q8  | A      | Phương thức thanh toán phổ biến nhất cho đơn hàng bị hủy: credit_card (28,452 đơn) |
| Q9  | A      | Kích thước có tỷ lệ trả hàng cao nhất: S (~5.65%) |
| Q10 | C      | Số lần trả góp có giá trị thanh toán trung bình cao nhất: 6 (~24,447) |

## Lưu ý

- Đảm bảo tất cả file CSV nằm trong thư mục `data/`
- Notebook được viết với đường dẫn dữ liệu mặc định là `data/`
- Có thể cần điều chỉnh đường dẫn dữ liệu nếu chạy trên Kaggle
