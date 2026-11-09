# TÀI LIỆU TỔNG QUAN LUỒNG MÔ PHỎNG SẠT LỞ ĐẤT VÀ AI CẢM BIẾN ẢO

**Phiên bản:** Tổng hợp theo các module M1--M5 và nhánh AI Virtual
Sensor Single Input (4.1--4.5) hiện tại\
**Phạm vi:** Mô phỏng số liệu tổng hợp (synthetic) trước, sau đó đánh
giá độc lập bằng dữ liệu bên ngoài và dữ liệu thí nghiệm vật lý.\
**Nguyên tắc quan trọng:** Các hệ số và ngưỡng mô phỏng hiện tại là giả
định phục vụ phát triển mô hình, không phải thông số địa kỹ thuật đã
được hiệu chuẩn hoặc ngưỡng cảnh báo sạt lở thực tế.

------------------------------------------------------------------------

## 1. Mục tiêu và kiến trúc tổng thể

Nghiên cứu được triển khai theo hướng xây dựng mô hình mô phỏng phản ứng
của đất dưới tác động mưa, mô tả các đại lượng trạng thái (độ ẩm, áp lực
nước lỗ rỗng), mô phỏng tín hiệu chuyển động, phân loại trạng thái tổng
hợp, phân tích thống kê và phát triển AI cảm biến ảo.

AI cảm biến ảo ở nhánh hiện tại có nhiệm vụ:

-   **Đầu vào duy nhất:** độ ẩm thể tích đất VWC.
-   **Đầu ra:** áp lực nước lỗ rỗng dương (PWP dương).
-   Huấn luyện và lựa chọn mô hình chỉ bằng dữ liệu synthetic do mô
    phỏng nội bộ tạo ra.
-   Sau khi chốt mô hình, dùng dữ liệu Trung Quốc/Zenodo làm tập đánh
    giá ngoài độc lập; không dùng tập này để huấn luyện, fine-tune,
    tuning siêu tham số hoặc lựa chọn mô hình.

### 1.1. Sơ đồ luồng tổng thể

``` mermaid
flowchart TD
    A[Thiết kế 2 mức mưa x 3 thời lượng] --> B[M1: Sinh mưa]
    B --> C[M2: Mô phỏng VWC]
    C --> D[M3: Độ ẩm trễ và PWP dương synthetic]
    D --> E[M4: Mô phỏng tín hiệu IMU và trích xuất đặc trưng]
    D --> F[M4-C2: Giả thuyết liên kết PWP - chuyển động]
    E --> G[M4-C1: Đồng bộ chuỗi dữ liệu]
    F --> H[M4-C4: Tích hợp dữ liệu]
    G --> H
    H --> I[M5: Phân loại trạng thái theo luật]
    I --> J[M5-B: State machine, hysteresis, dwell]
    J --> K[Dataset synthetic giai đoạn 1]
    K --> L[Phân tích giai đoạn 3]
    K --> M[4.1: Dataset AI - chỉ VWC]
    M --> N[4.2 Linear Regression]
    M --> O[4.3 Random Forest]
    M --> P[4.4 XGBoost]
    N --> Q[4.5 So sánh LOSO]
    O --> Q
    P --> Q
    Q --> R[Chọn Linear Regression]
    R --> S[Đánh giá ngoài bằng dữ liệu Trung Quốc]
    S --> T[Đối chiếu PWP dự đoán và PWP đo]
    U[Thí nghiệm hộp đất thực tế] --> V[Kiểm chứng xu hướng và đặc tính vật lý]
```

**Phân biệt hai nhánh:** Nhánh M1--M5 tạo dữ liệu mô phỏng để nghiên cứu
quy luật và xây dựng nhãn synthetic. Thí nghiệm hộp đất là nhánh thực
nghiệm độc lập để kiểm chứng xu hướng/giả thuyết. Dữ liệu thí nghiệm
không được chèn ngược vào bộ mô phỏng giai đoạn 1 theo quy trình đã
thống nhất.

------------------------------------------------------------------------

## 2. Thiết kế kịch bản mưa

Hai yếu tố đầu vào có thể chủ động điều khiển trong thí nghiệm được chọn
để tạo ma trận 2 × 3:

  Mã kịch bản          Cường độ mưa   Thời lượng
  ------------------ -------------- ------------
  LOW_20mmh_10min           20 mm/h      10 phút
  LOW_20mmh_20min           20 mm/h      20 phút
  LOW_20mmh_30min           20 mm/h      30 phút
  HIGH_40mmh_10min          40 mm/h      10 phút
  HIGH_40mmh_20min          40 mm/h      20 phút
  HIGH_40mmh_30min          40 mm/h      30 phút

Thiết lập mô phỏng hiện tại:

-   Bước thời gian: 1 phút.
-   Mưa bắt đầu tại phút thứ 10 để có đoạn nền trước mưa.
-   Sau khi mưa kết thúc, tiếp tục quan sát 120 phút.
-   VWC ban đầu: 0,15 m³/m³ (15%).
-   Các mức mưa/thời lượng là thiết kế kịch bản hiện tại; cần xác nhận
    bằng thiết bị tạo mưa thực tế khi làm thí nghiệm.

Các tham số thủy lực hiệu dụng như hệ số thấm/thoát nước không được khảo
sát như biến kịch bản ở phiên bản này. Chúng được giữ cố định trong mô
phỏng; chỉ thay đổi cường độ và thời lượng mưa.

------------------------------------------------------------------------

## 3. Module 1 -- Mô hình sự kiện mưa

### 3.1. Mục đích

M1 tạo chuỗi cường độ mưa theo thời gian cho từng kịch bản. Mưa có cường
độ không đổi trong khoảng sự kiện và bằng 0 trước/sau sự kiện.

### 3.2. Công thức cường độ mưa

Gọi:

-   (R(t)): cường độ mưa tại thời điểm (t), mm/h.
-   (R_0): cường độ được chọn cho kịch bản.
-   (t_s): thời điểm bắt đầu mưa.
-   (T_r): thời lượng mưa.

\[ R(t)=
```{=tex}
\begin{cases}
R_0, & t_s \le t < t_s+T_r\\
0, & \text{các thời điểm còn lại}
\end{cases}
```
\]

### 3.3. Lượng mưa tích lũy

Với bước thời gian (`\Delta `{=tex}t) tính bằng phút:

\[ P(t)=`\sum`{=tex}\_{i=1}\^{t}R_i`\frac{\Delta t}{60}`{=tex} \]

Trong đó (P(t)) có đơn vị mm.

### 3.4. Ước tính cường độ mưa từ nước hứng

Nếu đo được thể tích nước (V) (lít), diện tích hứng (A) (m²), thời gian
hứng (T) (phút):

\[ R=`\frac{60V}{AT}`{=tex}`\quad`{=tex}(`\mathrm{mm/h}`{=tex}) \]

Công thức dựa trên quan hệ 1 lít/m² tương đương 1 mm nước.

### 3.5. Tham số đang dùng

  Tham số                    Giá trị Ý nghĩa
  ---------------- ----------------- -----------------------
  Mưa thấp                   20 mm/h Mức cường độ thấp
  Mưa cao                    40 mm/h Mức cường độ cao
  Thời lượng         10, 20, 30 phút Ba mức thời lượng
  Bắt đầu mưa                10 phút Đoạn nền trước mưa
  Bước thời gian              1 phút Độ phân giải mô phỏng

**Đầu ra M1:** `rainfall_mm_h`, `cumulative_rainfall_mm`, `scenario_id`,
`time_min` và thông tin cấu hình từng kịch bản.

------------------------------------------------------------------------

## 4. Module 2 -- Mô hình phản ứng độ ẩm đất

### 4.1. Mục đích

M2 mô tả sự thay đổi VWC theo tác động mưa và quá trình thoát nước đơn
giản. Đây là mô hình bậc thấp dạng hiện tượng (reduced-order
phenomenological model), không phải mô hình thấm Richards và không giải
bài toán không gian.

### 4.2. Phương trình vi phân

\[ `\frac{d\theta}{dt}`{=tex} =
k\_{in}`\frac{R}{R_{ref}}`{=tex}(`\theta`{=tex}*{max}-`\theta`{=tex}) -
k_d(`\theta`{=tex}-`\theta`{=tex}*{min}) \]

Trong đó:

-   (`\theta`{=tex}): độ ẩm thể tích đất (m³/m³).
-   (R): cường độ mưa (mm/h).
-   (R\_{ref}): cường độ mưa tham chiếu (mm/h).
-   (k\_{in}): hệ số phản ứng/đưa nước vào hiệu dụng (phút⁻¹).
-   (k_d): hệ số thoát nước hiệu dụng (phút⁻¹).
-   (`\theta`{=tex}*{min}), (`\theta`{=tex}*{max}): giới hạn dưới/trên
    của VWC.

Số hạng thứ nhất biểu diễn xu hướng tăng ẩm do mưa; số hạng thứ hai biểu
diễn xu hướng giảm ẩm/thoát nước.

### 4.3. Rời rạc hóa Euler tường minh

\[ `\theta`{=tex}\_i= `\operatorname{clip}`{=tex}`\left[
\theta_{i-1}+\Delta t
\left(
k_{in}\frac{R_{i-1}}{R_{ref}}(\theta_{max}-\theta_{i-1})
-k_d(\theta_{i-1}-\theta_{min})
\right),
\theta_{min},\theta_{max}
\right]`{=tex}\]

`clip` giới hạn VWC trong khoảng
(\[`\theta`{=tex}*{min},`\theta`{=tex}*{max}\]).

### 4.4. Tham số phiên bản hiện tại

  Tham số                          Giá trị
  ------------------------- --------------
  (`\theta`{=tex}\_{min})       0,10 m³/m³
  (`\theta`{=tex}\_{max})       0,45 m³/m³
  (`\theta`{=tex}\_0)           0,15 m³/m³
  (R\_{ref})                       40 mm/h
  (k\_{in})                   0,025 phút⁻¹
  (k_d)                       0,003 phút⁻¹

Các tham số trên là giá trị giả định để phát triển mô phỏng. Không diễn
giải chúng như hệ số thấm hoặc thông số thủy lực đã đo của mẫu đất.
Trong thiết kế hiện tại, chúng được giữ cố định; không đưa vào ma trận
khảo sát kịch bản.

### 4.5. Đạo hàm/ tốc độ thay đổi độ ẩm

Code tính tốc độ thay đổi VWC bằng sai phân số:

\[ `\dot{\theta}`{=tex}(t)`\approx `{=tex}`\frac{d\theta}{dt}`{=tex} \]

Đại lượng này được lưu để mô tả diễn biến độ ẩm. Nó không được dùng làm
đầu vào của nhánh AI Single Input hiện tại.

### 4.6. Đổi đơn vị

\[ VWC(%)=100`\theta`{=tex} \]

**Đầu ra M2:** VWC dạng m³/m³ và %, tốc độ thay đổi độ ẩm, mưa và lượng
mưa tích lũy.

------------------------------------------------------------------------

## 5. Module 3 -- Mô hình độ ẩm trễ và áp lực nước lỗ rỗng dương

### 5.1. Mục đích

M3 nhận VWC từ M2, tạo một trạng thái độ ẩm trễ nhằm mô tả phản ứng
chậm, sau đó tính một đại lượng áp lực nước lỗ rỗng dương synthetic.

Đây là mô hình hiện tượng bậc thấp. Code không giải phương trình
Richards, không suy ra PWP từ ứng suất hữu hiệu/sức kháng cắt và không
mô hình hóa phân bố áp lực theo không gian. Giá trị kPa ở đây là thang
giả định, chưa phải phép đo PWP thực tế.

### 5.2. Trạng thái độ ẩm trễ

\[ `\frac{d\theta_d}{dt}`{=tex}=`\frac{\theta-\theta_d}{\tau}`{=tex} \]

Trong đó:

-   (`\theta`{=tex}): VWC hiện tại từ M2.
-   (`\theta`{=tex}\_d): VWC trễ.
-   (`\tau`{=tex}): hằng số thời gian trễ, phút.

Rời rạc hóa Euler:

\[ `\theta`{=tex}\_{d,i}= `\operatorname{clip}`{=tex}`\left[
\theta_{d,i-1}
+\Delta t\frac{\theta_{i-1}-\theta_{d,i-1}}{\tau},
0,\theta_{max}
\right]`{=tex}\]

Giá trị hiện tại: (`\tau=8`{=tex}) phút.

### 5.3. Hàm kích hoạt độ ẩm

\[ W_i= `\operatorname{clip}`{=tex} `\left`{=tex}(
`\frac{\theta_{d,i}-\theta_{critical}}`{=tex}
{`\theta`{=tex}*{max}-`\theta`{=tex}*{critical}}, 0,1 `\right`{=tex}) \]

-   Khi (`\theta`{=tex}*d`\leq`{=tex}`\theta`{=tex}*{critical}), (W=0).
-   Khi độ ẩm trễ tăng, (W) tăng.
-   Khi đạt hoặc vượt (`\theta`{=tex}\_{max}), (W=1).

Giá trị hiện tại: (`\theta`{=tex}*{critical}=0,18),
(`\theta`{=tex}*{max}=0,45) m³/m³.

### 5.4. Phương trình tạo và suy giảm PWP

\[ `\frac{du}{dt}`{=tex} = k_g
W`\left`{=tex}(1-`\frac{u}{u_{cap}}`{=tex}`\right`{=tex})-k_r u \]

Trong đó:

-   (u): áp lực nước lỗ rỗng dương synthetic (kPa).
-   (k_g): hệ số tạo áp lực giả định (kPa/phút).
-   (k_r): hệ số suy giảm áp lực (phút⁻¹).
-   (u\_{cap}): giới hạn áp lực mô phỏng (kPa).
-   (W): hàm kích hoạt độ ẩm không thứ nguyên.

Rời rạc hóa Euler:

\[ u_i=`\operatorname{clip}`{=tex}`\left[
u_{i-1}+\Delta t
\left(
k_g W_{i-1}\left(1-\frac{u_{i-1}}{u_{cap}}\right)
-k_r u_{i-1}
\right),
0,u_{cap}
\right]`{=tex}\]

Tham số hiện tại:

  Tham số                                Giá trị
  ------------------------------ ---------------
  (k_g)                            0,18 kPa/phút
  (k_r)                             0,008 phút⁻¹
  (u\_{cap})                               8 kPa
  (u_0)                                    0 kPa
  (`\tau`{=tex})                          8 phút
  (`\theta`{=tex}\_{critical})        0,18 m³/m³

Các tham số này đều là giả định mô phỏng. Không gọi PWP synthetic là
ground truth thực nghiệm.

### 5.5. Đầu ra M3

-   `delayed_moisture_vwc_m3_m3`
-   `delayed_moisture_percent`
-   `wetness_drive_0_1`
-   `pore_pressure_positive_kpa_synthetic`
-   `pressure_generation_rate_kpa_min`
-   `pressure_relaxation_rate_kpa_min`
-   `pressure_generation_active`

`pressure_generation_active` chỉ cho biết điều kiện kích hoạt tạo áp lực
theo công thức đã đặt được thỏa mãn; không phải nhãn sạt lở.

------------------------------------------------------------------------

## 6. Module 4 -- Mô phỏng chuyển động IMU và đặc trưng

Module 4 mô phỏng tín hiệu gia tốc kế/con quay hồi chuyển kiểu MPU6050,
sau đó trích xuất đặc trưng theo cửa sổ 10 giây. Đây là tín hiệu
synthetic, không phải tín hiệu đo từ ESP32/MPU6050 thực tế.

### 6.1. M4-A1 -- Tạo tín hiệu gia tốc và con quay

Tần số mô phỏng:

\[
f_s=10 `\mathrm{Hz}`{=tex},`\qquad `{=tex}`\Delta `{=tex}t=0,1 `\mathrm{s}`{=tex}
\]

Mô hình gia tốc gồm thành phần trọng lực chiếu theo góc nghiêng, chuyển
động động và nhiễu:

\[ a_x=g`\sin`{=tex}`\phi`{=tex}+a\_{x,dyn}+`\epsilon`{=tex}\_x \]

\[ a_y=a\_{y,dyn}+`\epsilon`{=tex}\_y \]

\[ a_z=g`\cos`{=tex}`\phi`{=tex}+a\_{z,dyn}+`\epsilon`{=tex}\_z \]

Trong đó (g=9,80665 `\mathrm{m/s^2}`{=tex}), (`\phi`{=tex}) là góc
nghiêng giả lập.

Tín hiệu động được tạo từ các thành phần sin có tần số minh họa 0,7 Hz
và 2 Hz, biên độ thay đổi theo trạng thái. Nhiễu gia tốc và gyro là
nhiễu Gaussian giả lập.

Con quay hồi chuyển mô tả vận tốc góc:

\[
`\omega`{=tex}*y(t)`\approx `{=tex}`\frac{d\phi}{dt}`{=tex}+`\text{thành phần dao động}`{=tex}+`\epsilon`{=tex}*`\omega`{=tex}
\]

Các giá trị vật lý sau đó được đổi sang số đếm RAW gần đúng của MPU6050:

\[
ACC\_{raw}=`\operatorname{round}`{=tex}`\left`{=tex}(`\frac{a}{g}`{=tex}`\times 16384`{=tex}`\right`{=tex})
\]

\[
GYRO\_{raw}=`\operatorname{round}`{=tex}(`\omega`{=tex}`\times131`{=tex})
\]

Các hệ số tương ứng cấu hình ±2g và ±250°/s. Đây là quy đổi mô phỏng
theo thang đo cảm biến, không phải hiệu chuẩn cảm biến vật lý.

### 6.2. M4-A2 -- Trích xuất đặc trưng cửa sổ 10 giây

Với (f_s=10) Hz và cửa sổ 10 giây:

\[ N=f_sT=10`\times10`{=tex}=100 `\text{mẫu/cửa sổ}`{=tex} \]

Trong code mô phỏng, thành phần trọng lực được trừ dựa trên góc nghiêng
tham chiếu đã biết:

\[ a\_{x,dyn}=a_x-g`\sin`{=tex}`\phi`{=tex} \]

\[ a\_{y,dyn}=a_y \]

\[ a\_{z,dyn}=a_z-g`\cos`{=tex}`\phi`{=tex} \]

Độ lớn gia tốc động:

\[ a\_{dyn}(i)= `\sqrt{a_{x,dyn}^2+a_{y,dyn}^2+a_{z,dyn}^2}`{=tex} \]

RMS gia tốc động:

\[ RMS\_{dyn}= `\sqrt{\frac{1}{N}\sum_{i=1}^{N}a_{dyn}(i)^2}`{=tex} \]

Trong file đặc trưng, RMS được đổi sang đơn vị (g) bằng cách chia cho
(g).

Phương sai:

\[ Var(a\_{dyn})=
`\frac{1}{N}`{=tex}`\sum`{=tex}*{i=1}^{N}a\_{dyn}(i)^2-
`\left`{=tex}(`\frac{1}{N}`{=tex}`\sum`{=tex}*{i=1}^{N}a\_{dyn}(i)`\right`{=tex})^2
\]

Độ lớn vận tốc góc:

\[ `\omega`{=tex}\_{mag}(i)=
`\sqrt{\omega_x(i)^2+\omega_y(i)^2+\omega_z(i)^2}`{=tex} \]

RMS vận tốc góc:

\[ RMS\_`\omega`{=tex}=
`\sqrt{\frac{1}{N}\sum_{i=1}^{N}\omega_{mag}(i)^2}`{=tex} \]

Ngoài ra code lưu độ lớn gia tốc trung bình và góc nghiêng trung bình
trong cửa sổ.

**Lưu ý triển khai thực tế:** Việc trừ trọng lực trong mô phỏng sử dụng
góc nghiêng tham chiếu có sẵn. Với MPU6050 thực, không có góc tham chiếu
đúng sẵn; cần xây dựng bộ lọc/ước lượng tư thế (ví dụ LPF, complementary
hoặc Kalman tùy thiết kế) và đánh giá sai số trước khi dùng công thức
tương tự.

### 6.3. M4-A3 -- Phân tích đặc trưng theo trạng thái

Các đặc trưng được tổng hợp theo bốn trạng thái giả lập:

-   STABLE
-   WATER_ACCUMULATION
-   DEFORMATION
-   SLIDING

Các phân tích gồm thống kê trung bình/độ lệch chuẩn, boxplot và mức độ
chồng lấn giữa các trạng thái. Mục đích là quan sát khả năng phân biệt
trạng thái của đặc trưng synthetic, không chứng minh các ngưỡng phân
loại ngoài thực địa.

### 6.4. M4-A4 -- Kiểm tra nội bộ

Kiểm tra tính hợp lệ của dữ liệu và đặc trưng, chẳng hạn:

-   Không có NaN/Inf.
-   RMS không âm.
-   Phương sai không âm.
-   Có đủ bốn trạng thái.
-   Kiểm tra cấu trúc và tính nhất quán các cửa sổ.

Các kiểm tra này xác nhận tính nhất quán của code/dữ liệu synthetic,
không phải xác nhận độ chính xác cảm biến thực.

### 6.5. M4-B -- Khảo sát độ nhạy chuyển động

M4-B thay đổi các biên độ và thời điểm chuyển trạng thái trong mô hình
IMU synthetic để xem đặc trưng thay đổi thế nào. Đây là phân tích độ
nhạy của giả định chuyển động, không phải hiệu chuẩn ngưỡng MPU6050
ngoài thực nghiệm.

### 6.6. M4-C1 -- Đồng bộ dữ liệu

M4-C1 đồng bộ chuỗi PWP synthetic của M3 với đặc trưng chuyển động M4-A2
theo `scenario_id` và `time_min`.

**C1 chỉ đồng bộ dữ liệu, không áp dụng quan hệ nhân quả PWP → chuyển
động.** Hai chuỗi ở C1 vẫn độc lập theo cách sinh dữ liệu tương ứng.

### 6.7. M4-C2 -- Giả thuyết liên kết PWP với chuyển động

C2 tạo một nhánh giả thuyết: PWP dương synthetic tăng thì chỉ số kích
hoạt chuyển động tăng theo.

Chuẩn hóa PWP thành mức kích hoạt đầu vào:

\[ D_i=`\operatorname{clip}`{=tex} `\left`{=tex}(
`\frac{u_i-u_{onset}}{u_{full}-u_{onset}}`{=tex},0,1 `\right`{=tex}) \]

Với:

-   (u\_{onset}=0,5) kPa.
-   (u\_{full}=3,2) kPa.

Đáp ứng trễ bậc nhất:

\[ `\alpha`{=tex}\_i=1-e\^{-`\Delta `{=tex}t/`\tau`{=tex}\_m} \]

\[ A_i=A\_{i-1}+`\alpha`{=tex}*i(D_i-A*{i-1}) \]

Trong đó (A_i`\in[0,1]`{=tex}) là `c2_motion_activation_0_1`,
(`\tau`{=tex}\_m=2) phút là độ trễ đáp ứng giả định.

Các đặc trưng chuyển động synthetic được ánh xạ từ (A):

\[ RMS\_{C2}=0,00052+0,002445A+0,017430A\^2 \]

\[ Gyro\_{C2}=0,060+0,18A+1,20A\^2 \]

\[ Tilt\_{C2}=0,15+4A+8A\^2 \]

Các công thức này là ánh xạ minh họa để tạo chuỗi đặc trưng có liên kết
với PWP; không phải định luật cấu thành đất đã hiệu chuẩn.

Trạng thái đáp ứng C2:

-   (A=0): NO_MOTION_RESPONSE.
-   (0\<A\<0,25): EARLY_RESPONSE.
-   (0,25`\le `{=tex}A\<0,75): DEFORMATION_RESPONSE.
-   (A`\ge0`{=tex},75): SLIDING_RESPONSE.

**Phân biệt:** `c2_motion_state` là trạng thái của nhánh giả thuyết C2.
Nó không đồng nhất với trạng thái tham chiếu M4-A1/A2.

### 6.8. M4-C3/C4 và tích hợp

-   C3 kiểm tra độ nhạy/độ bền của giả định liên kết, ví dụ thay đổi
    (u\_{onset}), (u\_{full}) và kiểm tra đáp ứng.
-   C4 tích hợp các chuỗi theo khóa kịch bản-thời gian, giữ riêng các
    nhánh tham chiếu M4-A1/A2 và giả thuyết C2.
-   M4-C4 tạo nền dữ liệu cho M5.

------------------------------------------------------------------------

## 7. Module 5 -- Phân loại trạng thái theo luật

### 7.1. Mục tiêu

M5 gán nhãn trạng thái theo thời gian cho từng kịch bản dựa trên các
biến synthetic: mưa, độ ẩm/kích hoạt ẩm, PWP và chỉ số chuyển động C2.

Bốn nhãn:

  Nhãn                 Ý nghĩa trong mô phỏng
  -------------------- ----------------------------
  STABLE               Trạng thái ổn định giả lập
  WATER_ACCUMULATION   Tích lũy nước/độ ẩm tăng
  DEFORMATION          Biến dạng giả lập
  SLIDING              Trượt giả lập

Đây là nhãn luật trên dữ liệu synthetic, không phải nhãn ground truth
của một vụ sạt lở thực tế.

### 7.2. M5-A -- Phân loại theo luật tức thời

M5-A áp dụng các điều kiện và thứ tự ưu tiên để gán trạng thái tại từng
thời điểm. Do chỉ xét điều kiện tại thời điểm hiện tại, nhãn có thể thay
đổi nhanh khi tín hiệu nằm gần ngưỡng.

Các điều kiện M5-A dùng các biến như wetness, áp lực, mưa và đặc
trưng/activation chuyển động theo cấu hình trong code. Không nên diễn
giải kết quả M5-A như xác suất sạt lở.

### 7.3. M5-B -- State machine, hysteresis và dwell time

M5-B dùng thứ tự chuyển trạng thái liền kề:

\[ STABLE`\rightarrow `{=tex}WATER_ACCUMULATION
`\rightarrow `{=tex}DEFORMATION`\rightarrow `{=tex}SLIDING \]

Có thể quay về trạng thái liền trước khi điều kiện thoát được thỏa mãn.

Điều kiện ướt:

\[ wet =
(R\>0)`\lor`{=tex}(W\>W\_{threshold})`\lor`{=tex}(u`\ge `{=tex}u\_{threshold})
\]

Điều kiện khô:

\[ dry =
(R`\le0`{=tex})`\land`{=tex}(W`\le `{=tex}W\_{threshold})`\land`{=tex}(u\<u\_{threshold})
\]

Ngưỡng hysteresis của activation:

  Chuyển trạng thái                           Ngưỡng
  ----------------------------- --------------------
  WATER → DEFORMATION (ON)        (A`\ge0`{=tex},25)
  DEFORMATION → WATER (OFF)                (A\<0,15)
  DEFORMATION → SLIDING (ON)      (A`\ge0`{=tex},75)
  SLIDING → DEFORMATION (OFF)              (A\<0,60)

Dwell time:

\[ T\_{dwell}=2 `\text{phút}`{=tex} \]

Một trạng thái ứng viên phải duy trì đủ 2 phút mới được xác nhận chuyển
trạng thái. Hysteresis sử dụng ngưỡng bật và tắt khác nhau để hạn chế
nhãn dao động quanh một ngưỡng.

### 7.4. M5-C/M5-D

-   M5-C phân tích điều kiện và thời điểm chuyển trạng thái, khoảng thời
    gian tồn tại của từng trạng thái và các điều kiện đi kèm.
-   M5-D kiểm toán tính nhất quán nội bộ của pipeline, khóa kịch
    bản-thời gian, nhãn, trạng thái và các đầu ra.
-   M5-D là kiểm tra logic/code, không phải đánh giá độ chính xác dự báo
    sạt lở.

### 7.5. Nhãn được chọn cho dataset tổng hợp

Trong dataset giai đoạn 1 hiện tại:

-   `event_label` là nhãn M5-B.
-   `event_label_code` là mã số tương ứng.
-   `event_label_method` ghi rõ phương pháp state
    machine/hysteresis/dwell.
-   `event_label_status` phải được hiểu là
    `SYNTHETIC_RULE_BASED_NOT_EXPERIMENTAL_GROUND_TRUTH`.

M4 reference motion và C2 hypothesis motion được lưu thành hai trường
khác nhau, không gộp làm một.

------------------------------------------------------------------------

## 8. Phân tích giai đoạn 3 trên dữ liệu synthetic

Giai đoạn 3 phân tích dataset giai đoạn 1, không tạo ra dữ liệu đo mới.

### 8.1. Tương quan

Pearson đo quan hệ tuyến tính; Spearman đo quan hệ đơn điệu dựa trên thứ
hạng. Tương quan không chứng minh quan hệ nhân quả. Dữ liệu chuỗi thời
gian có tự tương quan nên p-value cần được diễn giải thận trọng.

### 8.2. Phân tích độ trễ (lag)

Quy ước: lag dương nghĩa là X đi trước Y một khoảng thời gian. Phân tích
onset dựa trên ngưỡng xác định cho từng đại lượng. Độ trễ quan sát được
bao gồm thời gian tích lũy và thời điểm vượt ngưỡng; không đồng nghĩa
với độ trễ vật lý thuần của cảm biến.

### 8.3. Phân tích sự kiện trượt

Phân tích thời điểm xuất hiện nhãn SLIDING, các cửa sổ trước/trong/sau
sự kiện và giá trị các biến. Trong dataset synthetic hiện tại, chỉ một
kịch bản xuất hiện nhãn SLIDING theo M5-B. Vì vậy không thể suy rộng
thành ngưỡng trượt phổ quát.

### 8.4. Ngưỡng cảm biến

Ngưỡng M3-4 được tính từ trung điểm giữa median của các trạng thái liền
kề. Đây là ngưỡng chẩn đoán nội bộ trên nhãn synthetic, không phải
ngưỡng cảnh báo ngoài thực tế.

### 8.5. Chỉ số LRI

LRI tổng hợp ba thành phần: VWC, PWP dương synthetic và RMS chuyển động
C2. Mỗi thành phần được chuẩn hóa từng đoạn theo các ngưỡng trạng thái;
điểm được giới hạn trong khoảng 0--3.

\[ LRI=100`\times`{=tex}`\frac{S_{VWC}+S_{PWP}+S_{RMS}}{3}`{=tex} \]

Các mức phân loại hiện tại:

               LRI Mức
  ---------------- -----------
     0 đến dưới 25 LOW
    25 đến dưới 50 MODERATE
    50 đến dưới 75 HIGH
        75 đến 100 VERY_HIGH

Trọng số bằng nhau và ngưỡng phân loại là lựa chọn tạm thời. Do ngưỡng
được xây dựng từ nhãn M5-B và RMS C2 liên kết giả thuyết với PWP, LRI có
tính phụ thuộc/circular với pipeline synthetic. Không dùng LRI làm bằng
chứng xác nhận cảnh báo thực địa.

------------------------------------------------------------------------

## 9. AI cảm biến ảo -- Single Input (Giai đoạn 4)

### 9.1. Định nghĩa bài toán

Mục tiêu là ước lượng PWP dương từ một cảm biến độ ẩm đất:

\[ X=`\theta`{=tex}\_{VWC}(%) \]

\[ y=u\^+\_{PWP}(`\mathrm{kPa}`{=tex}) \]

Chỉ sử dụng VWC làm đầu vào. Không sử dụng delayed moisture, moisture
rate, lượng mưa, EP, IMU hoặc nhãn trạng thái làm feature trong nhánh
Single Input hiện tại.

### 9.2. 4.1 -- Tạo dataset AI

Nguồn: dataset synthetic giai đoạn 1.

Mỗi bản ghi gồm:

-   `scenario_id`
-   `time_min`
-   `moisture_vwc_percent` -- feature duy nhất
-   `pore_pressure_positive_kpa` -- target
-   metadata về nguồn dữ liệu và trạng thái target.

Kích thước: 900 mẫu, 6 kịch bản.

Phạm vi dữ liệu:

  Đại lượng                   Min         Max
  --------------------- --------- -----------
  VWC                     14,852%     29,989%
  PWP dương synthetic       0 kPa   3,207 kPa

Có 257 mẫu target bằng 0. Target là synthetic, không phải ground truth
thực nghiệm.

### 9.3. 4.2 -- Linear Regression

Mô hình:

\[ `\hat{u}`{=tex}=b_0+b_1`\theta`{=tex}\_{VWC} \]

Mô hình fit trên toàn bộ 900 mẫu sau đánh giá có hệ số:

\[ `\boxed{\hat{u}=-2,486935+0,161206\theta_{VWC}}`{=tex} \]

Đơn vị đầu ra kPa, VWC nhập theo phần trăm (ví dụ 20, không phải 0,20).

Kết quả LOSO pooled:

  Chỉ số                Giá trị
  ------------- ---------------
  MAE              0,572169 kPa
  RMSE             0,751757 kPa
  R²                   0,363608
  Dự đoán âm             54/900
  Min dự đoán     -0,156182 kPa
  Max dự đoán      2,050695 kPa

Lưu ý: hệ số trên được fit toàn bộ dataset để lưu mô hình triển khai. Dự
đoán LOSO được tạo bởi sáu mô hình fold riêng biệt, không phải từ mô
hình fit toàn bộ.

### 9.4. 4.3 -- Random Forest

Thiết lập hiện tại:

-   `n_estimators=300`
-   `max_depth=None`
-   `min_samples_split=2`
-   `min_samples_leaf=1`
-   `max_features=1.0`
-   `bootstrap=True`
-   `random_state=42`

Kết quả LOSO pooled:

  Chỉ số              Giá trị
  ------------ --------------
  MAE            0,820564 kPa
  RMSE           1,143738 kPa
  R²                -0,473067
  Dự đoán âm            0/900

### 9.5. 4.4 -- XGBoost

Thiết lập hiện tại:

-   `n_estimators=300`
-   `max_depth=4`
-   `learning_rate=0.05`
-   `subsample=0.8`
-   `colsample_bytree=1.0`
-   `objective=reg:squarederror`
-   `random_state=42`
-   `reg_lambda=1.0`

Kết quả LOSO pooled:

  Chỉ số              Giá trị
  ------------ --------------
  MAE            0,802707 kPa
  RMSE           1,123804 kPa
  R²                -0,422166
  Dự đoán âm            0/900

### 9.6. 4.5 -- So sánh và chọn mô hình

  Mô hình                  MAE (kPa)     RMSE (kPa)             R²
  ------------------- -------------- -------------- --------------
  Linear Regression     **0,572169**   **0,751757**   **0,363608**
  Random Forest             0,820564       1,143738      -0,473067
  XGBoost                   0,802707       1,123804      -0,422166

Theo tiêu chí MAE thấp, RMSE thấp và R² cao, Linear Regression được chọn
làm mô hình chính cho bước đánh giá ngoài.

Tuy nhiên, mô hình có 54 dự đoán âm. Vì PWP dương không thể âm theo định
nghĩa target, đây là hạn chế cần báo cáo. Không tự động cắt dự đoán âm
về 0 mà vẫn giữ nguyên các metric cũ. Nếu đánh giá biến thể có ràng buộc
không âm, phải đánh giá lại riêng bằng cùng quy trình kiểm định.

### 9.7. Phương pháp đánh giá LOSO

Có 6 kịch bản, mỗi lượt giữ lại toàn bộ một kịch bản làm test:

``` mermaid
flowchart TD
    A[900 mẫu, 6 kịch bản] --> B[Fold 1: giữ HIGH 40-10]
    A --> C[Fold 2: giữ HIGH 40-20]
    A --> D[Fold 3: giữ HIGH 40-30]
    A --> E[Fold 4: giữ LOW 20-10]
    A --> F[Fold 5: giữ LOW 20-20]
    A --> G[Fold 6: giữ LOW 20-30]
    B --> H[Train trên 5 kịch bản còn lại]
    C --> H
    D --> H
    E --> H
    F --> H
    G --> H
    H --> I[Gộp 900 dự đoán out-of-fold]
    I --> J[Tính MAE, RMSE, R²]
```

Các chỉ số:

\[
MAE=`\frac{1}{n}`{=tex}`\sum`{=tex}\_{i=1}\^{n}\|y_i-`\hat `{=tex}y_i\|
\]

\[ RMSE=`\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}`{=tex} \]

\[ R\^2=1-`\frac{\sum_i(y_i-\hat y_i)^2}`{=tex}
{`\sum`{=tex}\_i(y_i-`\bar `{=tex}y)\^2} \]

LOSO đánh giá khả năng dự đoán khi bỏ qua một kịch bản mưa. Tuy nhiên,
đây vẫn là dữ liệu synthetic và các mẫu theo thời gian có tự tương quan.
LOSO không tự động chứng minh khả năng tổng quát hóa sang loại đất, cảm
biến hoặc thí nghiệm khác.

------------------------------------------------------------------------

## 10. Đánh giá độc lập bằng dữ liệu Trung Quốc/Zenodo

Sau khi chọn mô hình trên dữ liệu synthetic, mới chuyển sang bộ dữ liệu
ngoài.

Các file người dùng đã cung cấp:

1.  `Gr_PWPraw_data10hz.csv` -- PWP thô 10 Hz.
2.  `Gr_meanfilt_pwpdata_0.1hz.csv` -- PWP lọc trung bình 0,1 Hz.
3.  `Granite_raw_data10hz.csv` -- dữ liệu Granite thô.
4.  `Gr_medifilt_epdata10hz.csv` -- EP lọc trung vị 10 Hz.
5.  `Gr_medifilt_vwcdata10hz.csv` -- VWC lọc trung vị 10 Hz.

Chưa được xác nhận đầy đủ ý nghĩa vật lý, đơn vị và ánh xạ vị trí của
từng kênh trong các file. Vì vậy, chưa được tự ý xem một cột là kPa hoặc
ghép kênh theo số thứ tự khi chưa có metadata/tài liệu.

### 10.1. Điều kiện bắt buộc trước khi test

1.  Xác định đơn vị VWC và PWP; có công thức hiệu chuẩn cảm biến nếu PWP
    đang là điện áp/tín hiệu thô.
2.  Xác định kênh VWC và PWP tương ứng vị trí đo hoặc quan hệ không gian
    phù hợp.
3.  Xác định timestamp gốc và cách tạo file PWP 0,1 Hz.
4.  Đồng bộ thời gian đúng cách; không chỉ dựa vào số dòng hoặc lấy mỗi
    100 mẫu nếu chưa xác minh pha/thời điểm bắt đầu.
5.  Kiểm tra giá trị âm của PWP và mốc áp suất tham chiếu.
6.  Chuyển VWC về đúng đơn vị % như input của mô hình.
7.  Chuyển PWP về kPa và xác định rõ có phải PWP dương theo cùng định
    nghĩa target hay không.
8.  Báo cáo phạm vi dữ liệu ngoài so với phạm vi huấn luyện; đánh dấu
    extrapolation/domain shift.

### 10.2. Quy tắc đánh giá độc lập

-   Giữ nguyên mô hình đã chọn và hệ số đã fit.
-   Không train lại, fine-tune hoặc tuning bằng dữ liệu Trung Quốc.
-   Không dùng dữ liệu Trung Quốc để chọn lại giữa LR/RF/XGBoost.
-   Tạo dự đoán từ VWC, sau đó so sánh với PWP đo thực tế cùng thời
    điểm.
-   Báo cáo MAE, RMSE, R², biểu đồ chuỗi thời gian và scatter dự
    đoán--thực đo.
-   Nếu đơn vị hoặc vị trí cảm biến chưa rõ, dừng đánh giá định lượng và
    ghi rõ dữ liệu chưa đủ điều kiện.

**Điểm quan trọng:** Dữ liệu Trung Quốc là tập đánh giá ngoài. Không đưa
dữ liệu này vào tập huấn luyện của mô hình Single Input trong quy trình
hiện tại.

------------------------------------------------------------------------

## 11. Vai trò của thí nghiệm hộp đất

Thí nghiệm vật lý là nhánh kiểm chứng độc lập:

-   Lắp đặt cảm biến độ ẩm, PWP (nếu có), IMU và thiết bị đo mưa.
-   Thực hiện 6 kịch bản mưa theo thiết kế.
-   Ghi nhận tín hiệu và các đặc trưng theo thời gian.
-   Đối chiếu xu hướng tăng ẩm, tích lũy áp lực và chuyển động với giả
    thuyết mô phỏng.
-   Đánh giá khả năng lặp lại và sự khác biệt giữa các lần thử.

Dữ liệu thí nghiệm không được đưa ngược vào dataset synthetic giai đoạn
1 theo quy trình hiện tại. Nếu sau này phát triển một mô hình hiệu
chỉnh/hiệu chuẩn riêng bằng dữ liệu thực nghiệm, cần định nghĩa thành
một nhánh nghiên cứu mới, phân tách rõ với kết quả synthetic ban đầu.

------------------------------------------------------------------------

## 12. Từ điển một số biến quan trọng

  ----------------------------------------------------------------------------------------
  Biến                                     Ý nghĩa                 Đơn vị
  ---------------------------------------- ----------------------- -----------------------
  `rainfall_mm_h`                          Cường độ mưa            mm/h

  `cumulative_rainfall_mm`                 Lượng mưa tích lũy      mm

  `moisture_vwc_m3_m3`                     Độ ẩm thể tích dạng tỷ  m³/m³
                                           lệ                      

  `moisture_percent_vwc`                   Độ ẩm thể tích          \%

  `delayed_moisture_vwc_m3_m3`             Độ ẩm trễ               m³/m³

  `wetness_drive_0_1`                      Mức kích hoạt ẩm chuẩn  0--1
                                           hóa                     

  `pore_pressure_positive_kpa_synthetic`   PWP dương synthetic     kPa giả định

  `dynamic_rms_g`                          RMS gia tốc động        g

  `gyro_rms_deg_s`                         RMS vận tốc góc         °/s

  `tilt_reference_mean_deg`                Góc nghiêng tham chiếu  độ
                                           trung bình              

  `c2_motion_activation_0_1`               Mức kích hoạt chuyển    0--1
                                           động nhánh C2           

  `event_label`                            Nhãn trạng thái M5-B    categorical

  `moisture_vwc_percent`                   Input AI Single Input   \%

  `pore_pressure_positive_kpa`             Target AI Single Input  kPa synthetic
  ----------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 13. Kết luận khoa học và giới hạn diễn giải

1.  M1--M3 là mô hình hiện tượng bậc thấp để tạo phản ứng mưa--độ
    ẩm--PWP synthetic; không thay thế mô hình thấm/địa kỹ thuật đầy đủ.
2.  M4 tạo tín hiệu IMU synthetic và đặc trưng chuyển động; cần thay thế
    phần giả lập bằng tín hiệu cảm biến thực khi triển khai.
3.  M4-C2 là giả thuyết liên kết PWP--chuyển động, không phải quan hệ
    nhân quả đã được chứng minh.
4.  M5 tạo nhãn theo luật trên dữ liệu synthetic; nhãn không phải ground
    truth sạt lở.
5.  Phân tích giai đoạn 3 và LRI mô tả cấu trúc nội bộ của mô phỏng;
    không phải đánh giá cảnh báo thực địa.
6.  AI Single Input hiện chọn Linear Regression dựa trên LOSO synthetic.
    Đây là lựa chọn nội bộ, chưa phải mô hình đã được xác thực ngoài.
7.  Chỉ sau khi xác nhận đơn vị, kênh cảm biến và đồng bộ dữ liệu Trung
    Quốc mới có thể thực hiện kiểm định ngoài định lượng.
8.  Thí nghiệm hộp đất là bước kiểm chứng độc lập, giúp đánh giá xu
    hướng và tính hợp lý vật lý của mô phỏng.

**Cách diễn đạt phù hợp trong báo cáo:** "Mô hình được phát triển và
đánh giá ban đầu trên dữ liệu mô phỏng tổng hợp. Khả năng chuyển giao
được kiểm tra tiếp trên dữ liệu độc lập và thí nghiệm vật lý." Tránh
viết "mô hình đã dự đoán chính xác sạt lở thực tế" khi chưa có bằng
chứng kiểm định tương ứng.

------------------------------------------------------------------------

## 14. Các tệp/code nền tảng đã dùng để tổng hợp

-   `rainfall_moisture_simulation_v2(2).py` -- Module 1 và 2.
-   `pore_pressure_module_3(2).py` -- Module 3.
-   `module_4_motion_model_M5D_final.py` -- Module 4 và Module 5.
-   `Giai_doan_1_Synthetic_Dataset_6_kich_ban.csv` -- dataset synthetic
    tích hợp.
-   `Giai_doan_4_1_Single_Input_Dataset.csv` -- dataset AI Single Input.
-   Kết quả 4.2 Linear Regression, 4.3 Random Forest, 4.4 XGBoost và 4.5
    Model Comparison.

Tài liệu này phản ánh cấu hình và kết quả hiện tại của dự án; nếu code
hoặc tham số được thay đổi, cần cập nhật lại các phương trình, bảng tham
số và metric tương ứng.
