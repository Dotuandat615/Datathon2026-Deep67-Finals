# Datathon2026-Deep67-Finals

## Tổng quan dự án

Kho lưu trữ này chứa bài làm vòng final của Datathon 2026 từ nhóm Deep67, được tổ chức thành hai phần chính: phân tích dữ liệu khám phá (EDA) và pipeline hệ thống gợi ý (recommender system). [Notebook EDA](notebooks/contact_factor_analysis.ipynb) tập trung vào việc hiểu conversion funnel, hành vi người dùng, và các yếu tố ở cấp listing ảnh hưởng đến contact propensity, trong khi pipeline recommender system nạp các bộ dữ liệu parquet đã làm sạch, xây dựng aggregate ở mức user và item, sinh recall candidates từ nhiều nguồn, huấn luyện mô hình learning-to-rank với validation theo thời gian, và xuất file submission theo đúng định dạng yêu cầu.

Phần triển khai chính của recommender nằm trong notebook [recsys_5_fixed.ipynb](notebooks/recsys_5_fixed.ipynb). Notebook này tuân theo quy trình leakage-safe, trong đó tất cả feature đều được tính đến trước feature cutoff date, và sử dụng một multi-stage ranking stack được tối ưu cho recall và chất lượng xếp hạng cuối cùng.

## Cấu trúc repository

- [notebooks/contact_factor_analysis.ipynb](notebooks/contact_factor_analysis.ipynb): Notebook EDA chính, tập trung vào các yếu tố ảnh hưởng đến xác suất phát sinh liên hệ.
- [notebooks/eda_datathon.ipynb](notebooks/eda_datathon.ipynb): Notebook EDA nhẹ để profiling dữ liệu và khám phá baseline trước khi training model.
- [notebooks/recsys_5_fixed.ipynb](notebooks/recsys_5_fixed.ipynb): Pipeline end-to-end (feature engineering, candidate generation, training, scoring, submission).
- [notebooks/recsys_5_market_health_compare.ipynb](notebooks/recsys_5_market_health_compare.ipynb): Notebook so sánh market health.
- [notebooks/recsys_5_market_health_compare_2.ipynb](notebooks/recsys_5_market_health_compare_2.ipynb): Notebook so sánh market health phiên bản khác.
- [notebooks/05_marketplace_health_tradeoff.ipynb](notebooks/05_marketplace_health_tradeoff.ipynb): Notebook về trade-off của marketplace health.
- datathon_2026_processed/: Dữ liệu parquet đã làm sạch và test users.
- marketplace_health_tradeoff_assets/: Tài nguyên được tạo ra cho notebook marketplace health trade-off.
- output/: File `submission.csv` được sinh ra và `sample_submission.csv` được cung cấp.
- report_recsys_5_assets/: Tài nguyên báo cáo và các metric tổng hợp.
- save_parquet/: Các file parquet cache tạm và file DuckDB temp tùy chọn.
- src/: Các helper Python dùng chung cho logic health và reranking.
- tools/: Các tiện ích tạo notebook và tạo report.

### Sơ đồ cấu trúc thư mục

```bash
Datathon2026-Deep67-Finals/
├─ README.md
├─ datathon_2026_processed/
│  ├─ train_clean/
│  │  ├─ dim_listing/
│  │  │  └─ dim_listing_cleaned.parquet
│  │  ├─ fact_listing_snapshot/
│  │  │  └─ fact_listing_snapshot_cleaned.parquet
│  │  ├─ fact_user_ad_interactions/
│  │  │  └─ *.parquet
│  │  └─ fact_user_events/
│  │     └─ **/*.parquet
│  └─ test/
│     └─ test_users.parquet
├─ marketplace_health_tradeoff_assets/
├─ notebooks/
│  ├─ contact_factor_analysis.ipynb
│  ├─ 05_marketplace_health_tradeoff.ipynb
│  ├─ eda_datathon.ipynb
│  ├─ recsys_5_fixed.ipynb
│  ├─ recsys_5_market_health_compare.ipynb
│  ├─ recsys_5_market_health_compare_2.ipynb
│  └─ outputs/
├─ output/
│  ├─ sample_submission.csv
│  └─ submission.csv
├─ report_recsys_5_assets/
└─ save_parquet/
 └─ duckdb_temp/
├─ src/
│  ├─ __init__.py
│  ├─ health.py
│  └─ rerank.py
└─ tools/
 ├─ build_marketplace_health_notebook.py
 └─ generate_report_assets.py
```

## 1. Trực quan hóa và phân tích dữ liệu

Notebook chính cho phần này là [contact_factor_analysis.ipynb](notebooks/contact_factor_analysis.ipynb), trong đó đi sâu vào toàn bộ quá trình phân tích xác suất phát sinh liên hệ trên Chợ Tốt BĐS.

Phần này tập trung vào việc hiểu dữ liệu và xác định các yếu tố ảnh hưởng đến khả năng một tin đăng tạo ra hard contact. Notebook được xây theo hướng storytelling with data: bắt đầu từ bức tranh tổng quan của phễu chuyển đổi, sau đó đi lần lượt vào hành vi người dùng, đặc điểm listing, và cuối cùng là tổng hợp root cause cùng hướng hành động.

Điểm nổi bật nhất của phân tích là phễu chuyển đổi cho thấy lượng traffic không thiếu, nhưng thất thoát rất lớn ở bước Pageview -> Hard Contact. Trong notebook, khoảng 65.5 triệu pageviews chỉ tạo ra khoảng 5.5 triệu hard contacts, tương đương mức rơi hơn 91%. Từ đó, trọng tâm của phần EDA không phải là “có bao nhiêu lượt xem”, mà là “yếu tố nào làm một lượt xem có khả năng chuyển thành liên hệ cao hơn”.

Nội dung chính bao gồm:

- Xác định điểm rơi lớn nhất trong phễu chuyển đổi, với trọng tâm là bước Pageview -> Hard Contact, đồng thời đối chiếu với các nấc như chat và purchase proxy để thấy hành trình chuyển đổi thực tế.
- So sánh funnel giữa hai nhóm hành vi chính là Mua Bán và Cho Thuê để thấy khác biệt về động cơ, mức độ cấp bách và khả năng phát sinh liên hệ. Nhóm Cho Thuê thường mang tính nhu cầu ngắn hạn hơn, trong khi Mua Bán có chu kỳ cân nhắc dài hơn.
- Phân tích temporal patterns theo ngày trong tuần, theo giờ trong ngày, và theo giai đoạn thời gian để tìm ra nhịp truy cập tự nhiên, các đỉnh hoạt động, và thời điểm mà hard contact rate thường cải thiện.
- Khảo sát hành vi người dùng qua session depth, dwell time, login type, device, và bối cảnh tương tác trước khi phát sinh lead để hiểu người dùng nào có xu hướng chuyển đổi cao hơn.
- Phân tích các yếu tố của listing như category, price bucket, seller type, chất lượng tin đăng, độ mới, vị trí, số ảnh, và các tín hiệu content khác nhằm xác định tin đăng nào dễ được liên hệ hơn.
- Quan sát thêm các hiệu ứng về freshness, long-tail exposure, và mức độ phân bổ hiển thị để thấy không chỉ nội dung tin đăng mà cả cách phân phối tin đăng cũng ảnh hưởng đến khả năng phát sinh contact.
- Tổng hợp root cause để chỉ ra nhóm yếu tố nào làm tăng hoặc làm giảm hard contact rate, từ đó phân loại rõ đâu là vấn đề về nhu cầu người dùng, đâu là vấn đề về chất lượng listing, và đâu là vấn đề về hiển thị/phân phối.
- Từ các phát hiện trên, rút ra gợi ý cải thiện thực tiễn cho sản phẩm và cho việc ưu tiên phân phối tin đăng, thay vì chỉ dừng ở mức mô tả dữ liệu.

## 2. Mô hình gợi ý (Recommender System)

### Sơ đồ luồng dữ liệu và vị trí file

```bash
datathon_2026_processed/
 ├─ train_clean/dim_listing/*.parquet -----------┐
 ├─ train_clean/fact_listing_snapshot/*.parquet ──┼─> [recsys_5_fixed.ipynb]
 ├─ train_clean/fact_user_ad_interactions/*.parquet
 ├─ train_clean/fact_user_events/**/*.parquet ----┘
 └─ test/test_users.parquet ----------------------┐
                                                  │
 [recsys_5_fixed.ipynb] builds:                   │
  - user aggregates                               │
  - item aggregates                               │
  - user preferences                              │
  - recall candidates (pop, history, i2i, ALS)    │
  - pair features + hard negatives                │
  - LightGBM ranker                               │
  - test scoring + post-ranking                   │
                                                  │
 writes outputs:                                  │
  - save_parquet/*.parquet (optional cache) <----┘
  - output/submission.csv (final)
```

### Dữ liệu đầu vào

Notebook này kỳ vọng cấu trúc dữ liệu đã làm sạch nằm trong `datathon_2026_processed/`:

- `train_clean/dim_listing/dim_listing_cleaned.parquet`
- `train_clean/fact_listing_snapshot/fact_listing_snapshot_cleaned.parquet`
- `train_clean/fact_user_ad_interactions/*.parquet`
- `train_clean/fact_user_events/**/**/*.parquet`
- `test/test_users.parquet`

Các đường dẫn được cấu hình ở đầu notebook (hiện đang dùng Windows absolute paths). Nếu bạn chuyển vị trí dataset, hãy cập nhật `BASE_DATA_DIR` trong cell code đầu tiên của [notebooks/recsys_5_fixed.ipynb](notebooks/recsys_5_fixed.ipynb).

### Pipeline end-to-end (toàn bộ workflow)

1.**Môi trường và cấu hình**

- Xác định các cửa sổ thời gian: feature cutoff, training window, validation window, và prediction window.
- Cấu hình DuckDB và các đường dẫn cache tùy chọn.
- Đăng ký các dataset parquet dưới dạng DuckDB views.

2.**Tập người dùng mục tiêu**

- Xây dựng tập mục tiêu bằng cách kết hợp toàn bộ test users với một tập con người dùng train được sample.
- Tất cả bước aggregate feature phía sau đều chỉ được tính trên tập người dùng này.

3.**Aggregate ở mức user (tính đến feature cutoff)**

- Tổng số views, positive events, positive rate, thời gian dwell trung bình, số ngày hoạt động gần nhất, login type.
- Các aggregate bổ sung từ ad interaction nếu có, như ad views và ad leads.

4.**Aggregate ở mức item (tính đến feature cutoff)**

- Views và contacts tích lũy.
- Xu hướng 7 ngày gần nhất và contact rate gần đây.
- Metadata của listing như category, city, district, price bucket, beds, baths, area, images.
- Độ tuổi của item tại thời điểm cutoff và feature lead velocity nếu có dữ liệu contact interactions.

5.**Mô hình hóa sở thích người dùng**

- Xác định category, city, district và price bucket ưa thích của từng user dựa trên positive events và pageviews.

6.**Sinh candidate (giai đoạn recall)**

- Global popularity (top item toàn cục).
- Top item theo category, city, district, và price bucket.
- Lịch sử người dùng (recent pageviews).
- Item-to-item co-occurrence trong các session windows.
- ALS collaborative filtering (tùy chọn nếu đã cài `implicit`).
- Loại bỏ các positive interactions trước đó và khử trùng lặp.
- Bổ sung để đảm bảo số lượng candidate tối thiểu cho mỗi user và áp dụng recall boost cuối để ổn định kích thước pool.

7.**Feature engineering cho từng cặp user-item**

- Join các aggregate ở mức user, item và user-item.
- Tạo các match feature giữa sở thích user và thuộc tính item.
- Thêm freshness và recency weights.

8.**Huấn luyện và validation (chia theo thời gian)**

- Train positives: các sự kiện nằm trong training window.
- Validation positives: các sự kiện nằm trong validation window.
- Tạo hard negatives từ các item đã xem nhưng không positive.
- Downsample soft negatives và giới hạn số lượng theo từng user để an toàn bộ nhớ.
- Huấn luyện LightGBM LambdaRank và đánh giá Recall@10 cùng NDCG@10.

9.**Scoring và hậu xử lý**

- Chấm điểm candidate cho test users theo từng chunk để kiểm soát bộ nhớ.
- Áp dụng các rule hậu xếp hạng: freshness boost, category diversity penalty, và private-agent fairness.
- Dùng cold-user fallback với top item toàn cục hoặc theo city.

10.**Xuất submission**

- Đảm bảo khớp với `output/sample_submission.csv`.
- Lưu file cuối cùng `output/submission.csv` với các cột `ID`, `user_id`, `rank`, `item_id`.

## Cách chạy

1. Mở [notebooks/recsys_5_fixed.ipynb](notebooks/recsys_5_fixed.ipynb) trong VS Code.
2. Cập nhật `BASE_DATA_DIR` nếu đường dẫn dataset của bạn khác.
3. Chạy các cell theo thứ tự từ trên xuống dưới.
4. Submission cuối cùng sẽ được ghi ra `output/submission.csv`.

## Ghi chú và khuyến nghị

- Bật caching bằng cách đặt `USE_CACHE = True` sau lần chạy đầy đủ đầu tiên.
- Candidate generation bằng ALS yêu cầu `implicit` (cài bằng `pip install implicit`).
- Huấn luyện LightGBM yêu cầu `lightgbm` và `scikit-learn` (cài bằng `pip install lightgbm scikit-learn`).
- Với máy có bộ nhớ hạn chế, nên giữ các giới hạn sampling và validation downsampling đang bật.
