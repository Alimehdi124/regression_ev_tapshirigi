from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

qovluq = Path(__file__).resolve().parent
sutunlar = ["cicek_say", "temperatur", "rutubet"]
yeni_gun = pd.DataFrame([[180, 26, 55]], columns=sutunlar)
neticeler = []

for say in [10, 20, 50]:
    ad = f"bal_{say}"
    melumat = pd.read_csv(qovluq / "data" / f"{ad}.csv")
    x = melumat[sutunlar]
    y = melumat["bal_kq"]

    linear = LinearRegression()
    linear.fit(x, y)
    linear_proqnoz = linear.predict(x)
    print(f"\n{ad} | Multiple Linear Regression")
    print("w:", dict(zip(sutunlar, linear.coef_.round(5))))
    print("b:", round(linear.intercept_, 5))
    print("Yeni gün:", round(linear.predict(yeni_gun)[0], 3), "kq")
    neticeler.append([ad, "Multiple Linear", mean_absolute_error(y, linear_proqnoz), mean_squared_error(y, linear_proqnoz), r2_score(y, linear_proqnoz)])

    cevirici = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = cevirici.fit_transform(x)
    yeni_poly = cevirici.transform(yeni_gun)
    polynomial = LinearRegression()
    polynomial.fit(x_poly, y)
    polynomial_proqnoz = polynomial.predict(x_poly)
    print(f"\n{ad} | Polynomial Regression")
    print("Əmsallar:", dict(zip(cevirici.get_feature_names_out(sutunlar), polynomial.coef_.round(5))))
    print("b:", round(polynomial.intercept_, 5))
    print("Yeni gün:", round(polynomial.predict(yeni_poly)[0], 3), "kq")
    neticeler.append([ad, "Polynomial", mean_absolute_error(y, polynomial_proqnoz), mean_squared_error(y, polynomial_proqnoz), r2_score(y, polynomial_proqnoz)])

cedvel = pd.DataFrame(neticeler, columns=["Fayl", "Model", "MAE", "MSE", "R²"])
print("\nMetrics cədvəli")
print(cedvel.round(4).to_string(index=False))
cedvel.to_csv(qovluq / "metrics.csv", index=False)
