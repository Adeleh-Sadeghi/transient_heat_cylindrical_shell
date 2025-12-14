import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tabulate import tabulate

# تعریف پارامترهای مسئله
r1, r2 = 0.1, 0.2   # شعاع داخلی و خارجی (برحسب متر) 
L = 0.5             # طول استوانه (برحسب متر)
Nr, Nz = 20, 20     # تعداد نقاط شبکه در جهت r و z
dr = (r2 - r1) / (Nr - 1)  # فاصله مش در جهت r
dz = L / (Nz - 1)          # فاصله مش در جهت z

# پارامترهای ماده
rho = 7800  # چگالی (برحسب kg/m^3)
cp = 500    # ظرفیت حرارتی ویژه (برحسب J/kg.C)
k = 50      # هدایت حرارتی (برحسب W/m.C)
alpha = k / (rho * cp)  # ضریب نفوذ حرارتی

# پارامترهای زمان
dt = 0.01  # گام زمانی
Nt = 500    # تعداد گام‌های زمانی

# ایجاد مش در راستای r و z
r = np.linspace(r1, r2, Nr)
z = np.linspace(0, L, Nz)
R, Z = np.meshgrid(r, z, indexing='ij')

# مقداردهی اولیه دما
T = np.ones((Nr, Nz)) * 50  # مقدار اولیه تقریبی
T_new = np.copy(T)
error_table = []

# اعمال شرایط مرزی
T[0, :] = 100   # در r = r1
T[-1, :] = 20   # در r = r2
T[:, 0] = 100   # در z = 0
T[:, -1] = 40   # در z = L

# حل عددی با روش تفاضل متناهی مرتبه چهارم در زمان
for t in range(2, Nt-2):  # از گام‌های زمانی از t=2 تا t=Nt-2 برای استفاده از نقاط قبل و بعد
    for i in range(2, Nr - 2):  # برای استفاده از تفاضلات چهارم باید از i=2 شروع کنیم
        for j in range(2, Nz - 2):  # برای استفاده از تفاضلات چهارم باید از j=2 شروع کنیم
            r_i = r[i]
            T_new[i, j] = T[i, j] + alpha * dt * (
                (-T[i+2, j] + 16*T[i+1, j] - 30*T[i, j] + 16*T[i-1, j] - T[i-2, j]) / (12 * dr**2) +
                (-T[i, j+2] + 16*T[i, j+1] - 30*T[i, j] + 16*T[i, j-1] - T[i, j-2]) / (12 * dz**2)
            )
    
    # به‌روزرسانی دما
    T[:, :] = T_new[:, :]
    
    # محاسبه خطا
    max_change = np.max(np.abs(T_new - T))
    error_table.append([t, max_change])
    if max_change < 1e-6:
        print(f"Solution Converged at time step {t}!")
        break

# انتخاب گام‌های زمانی برای نمایش (ابتدا، انتها، دو تایم میانه)
time_steps_to_plot = [0, Nt//3, 2*Nt//3, Nt-1]

# تنظیم اندازه شکل
fig, axes = plt.subplots(4, 2, figsize=(12, 12))

for idx, t in enumerate(time_steps_to_plot):
    T = np.copy(T_new)  # بازگشت به آخرین حالت دما برای نمایش
    # نمایش 2D
    ax = axes[idx, 0]
    c = ax.contourf(z, r, T, 20, cmap='jet')
    ax.set_xlabel("z (m)")
    ax.set_ylabel("r (m)")
    ax.set_title(f"2D Temperature Distribution at Time Step {t}")
    fig.colorbar(c, ax=ax, label="Temperature (°C)")

    # نمایش 3D
    ax3d = axes[idx, 1]  # درست کردن نمای 3D
    ax3d = fig.add_subplot(4, 2, idx*2+2, projection='3d')  # ایجاد زیرنمودار سه‌بعدی
    surf = ax3d.plot_surface(Z, R, T, cmap='jet')
    ax3d.set_xlabel("z (m)")
    ax3d.set_ylabel("r (m)")
    ax3d.set_zlabel("Temperature (°C)")
    ax3d.set_title(f"3D Temperature Distribution at Time Step {t}")
    fig.colorbar(surf, ax=ax3d, label="Temperature (°C)")

# نمایش
plt.tight_layout()
plt.show()

# نمایش جدول خطا با کادربندی
error_df = pd.DataFrame(error_table, columns=["Time Step", "Max Change"])
print("\nError Table:")
print(tabulate(error_df, headers="keys", tablefmt="grid"))
