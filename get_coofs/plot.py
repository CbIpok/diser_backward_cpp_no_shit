import json
import matplotlib.pyplot as plt

# === Вариант A: JSON в файле ===
with open('case_statistics_hd_y_basis_4Tokai_most_o.json', 'r', encoding='utf-8') as f:
     data = json.load(f)



# 1) Извлекаем точки из ключей вида "[x,y]"
points = []
for key in data.keys():
    if key.startswith('[') and ',' in key:
        x_str, y_str = key.strip('[]').split(',')
        points.append((float(x_str), float(y_str)))

# 2) Функция для вычисления выпуклой оболочки (Monotone Chain)
def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def convex_hull(pts):
    pts = sorted(set(pts))
    if len(pts) <= 1:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

hull = convex_hull(points)

# 3) Подготовка точек для отрисовки замкнутого многоугольника
hull_loop = hull + [hull[0]]
hx, hy = zip(*hull_loop)
px, py = zip(*points)

# 4) Рисуем
plt.figure(figsize=(8,6))
plt.fill(hx, hy, alpha=0.3, edgecolor='blue', linewidth=2, label='Convex Hull')
plt.scatter(px, py, color='red', label='Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Область, образованная точками')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
