# Part 2 - EDA Narrative (Tiếng Việt)

## Descriptive — What happened?
- **Data quality gate**: Revenue ghép từ transaction khớp `sales.csv` (MAE=0.000000, max abs err=0.000000); do đó metric doanh thu dùng trong EDA có tính nhất quán cao.
- **Mùa vụ rõ rệt**: tháng 5 có seasonality index 1.55, trong khi tháng 12 là 0.60; chênh lệch peak/trough xấp xỉ 2.60x.
- **Doanh thu tập trung cao**: danh mục lớn nhất là **Streetwear** (share=79.92%), top-2 danh mục chiếm 95.11% tổng doanh thu, hàm ý rủi ro concentration.

## Diagnostic — Why did it happen?
- Áp lực tồn kho có liên hệ với tăng trưởng tháng kế tiếp: corr(stockout_days, next_growth)=-0.093, corr(fill_rate, next_growth)=0.093.
- So sánh theo quartile tồn kho: nhóm **Q1-low stockout** có tăng trưởng kế tiếp trung bình 5.03%, trong khi **Q4-high stockout** chỉ -0.89%.
- Discount depth không tự động cải thiện return rate: nhóm **No promo** return 6.38% vs nhóm **>15%** return 6.28%; cần tối ưu theo mục tiêu margin thay vì tăng độ sâu giảm giá đại trà.

## Predictive — What is likely to happen?
- Dự phóng 6 tháng (trend + seasonality) cho thấy doanh thu trung bình kịch bản **base** khoảng 103,111,855/tháng.
- Biên kịch bản: **optimistic** ~118,806,834/tháng và **conservative** ~87,416,877/tháng, phản ánh biến động mùa vụ quan sát trong lịch sử.
- Hàm ý vận hành: cần chuẩn bị tồn kho theo tháng đỉnh của từng category (không dùng một lịch mua hàng đồng nhất cho toàn bộ danh mục).

## Prescriptive — What should we do?
- Ưu tiên hành động #1: **GenZ - tháng 6** | stockout_days=1.35 | score=2.08.
  Hành động: Pre-build inventory 4-6 weeks before peak month; tighten safety stock and supplier lead-time monitoring
- Ưu tiên hành động #2: **Streetwear - tháng 4** | stockout_days=1.41 | score=1.37.
  KPI đề xuất: Reduce stockout_days by >=20%; keep fill_rate >=96%
- Quy tắc điều hành đề xuất: với category-tháng có priority score cao, khóa kế hoạch pre-build trước 4-6 tuần; với score thấp, giữ chính sách hiện tại để tránh overstock.

## Mapping Từ Figure Sang Insight
- Fig01: xu hướng doanh thu dài hạn và dao động mùa vụ.
- Fig02: heatmap mùa vụ theo năm-tháng, xác định tháng đỉnh/đáy.
- Fig03: Pareto concentration theo category.
- Fig04: profile mùa vụ khác biệt theo từng category.
- Fig05: mối liên hệ giữa stockout pressure và tăng trưởng kế tiếp.
- Fig06: hotspot month-category cần xử lý tồn kho.
- Fig07: diagnostic phụ cho chiến lược discount depth.
- Fig08: 3 kịch bản doanh thu 6 tháng để hỗ trợ planning.