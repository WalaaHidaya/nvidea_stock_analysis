# Charger vos données
data <- read.csv("C:/Users/walaa/Downloads/NVDA.csv")
data$Date <- as.Date(data$Date)

# Filtrer à partir de la date souhaitée (ex: 2000-01-03)
row_num <- which(data$Date == "2000-01-03")
data <- data[row_num:nrow(data), ]

# Convertir en série temporelle le prix de clôture ajusté
ts_data <- ts(data$Adj.Close, start = c(2000, 1), frequency = 251)

# Visualisation
plot(ts_data)

# Filtrer pour une autre fenêtre (ex: à partir de 2020-01-02)
row_num <- which(data$Date == "2020-01-02")
data <- data[row_num:nrow(data), ]
ts_data <- ts(data$Adj.Close, start = c(2020, 1), frequency = 251)
plot(ts_data)

# Division de l’échantillon (90% train, 10% test)
train_size <- floor(0.9 * nrow(data))
train_data <- data[1:train_size, ]
test_data <- data[(train_size + 1):nrow(data), ]

# Création des vecteurs ts pour Adj.Close
train_ts <- ts(train_data$Adj.Close, start = c(as.numeric(format(train_data$Date[1], "%Y")),
                                               as.numeric(format(train_data$Date[1], "%j"))),
               frequency = 251)
test_ts <- ts(test_data$Adj.Close, start = c(as.numeric(format(test_data$Date[1], "%Y")),
                                             as.numeric(format(test_data$Date[1], "%j"))),
              frequency = 251)

# ============================================================================
# ANALYSE EXPLORATOIRE COMPLÈTE - SÉRIE TEMPORELLE NVIDIA (NVDA)
# ============================================================================

# Chargement des packages nécessaires
library(forecast)
library(tseries)
library(ggplot2)
library(gridExtra)
library(TSA)

# ============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================

# Charger les données
data <- read.csv("C:/Users/walaa/Downloads/NVDA.csv")
data$Date <- as.Date(data$Date)

# Filtrer à partir de 2020-01-02 (comme dans votre code)
row_num <- which(data$Date == "2020-01-02")
data <- data[row_num:nrow(data), ]

# Création de la série temporelle
ts_data <- ts(data$Adj.Close, start = c(2020, 1), frequency = 251)

# Division train/test (90%/10%)
train_size <- floor(0.9 * nrow(data))
train_data <- data[1:train_size, ]
test_data <- data[(train_size + 1):nrow(data), ]

train_ts <- ts(train_data$Adj.Close, 
               start = c(as.numeric(format(train_data$Date[1], "%Y")),
                         as.numeric(format(train_data$Date[1], "%j"))),
               frequency = 251)

test_ts <- ts(test_data$Adj.Close, 
              start = c(as.numeric(format(test_data$Date[1], "%Y")),
                        as.numeric(format(test_data$Date[1], "%j"))),
              frequency = 251)

# ============================================================================
# 2. VISUALISATION DE LA SÉRIE COMPLÈTE
# ============================================================================

par(mfrow = c(2, 2), mar = c(4, 4, 3, 2))

# Série temporelle complète
plot(ts_data, main = "Prix Ajusté NVIDIA (2020-2024)", 
     ylab = "Prix ($)", xlab = "Temps", col = "darkblue", lwd = 1.5)
grid()

# Train vs Test
plot(train_ts, main = "Division Train/Test", 
     ylab = "Prix ($)", xlab = "Temps", col = "blue", lwd = 1.5)
lines(test_ts, col = "red", lwd = 1.5)
legend("topleft", legend = c("Train (90%)", "Test (10%)"), 
       col = c("blue", "red"), lty = 1, lwd = 2)
grid()

# Log-transformation (pour stabiliser la variance)
plot(log(train_ts), main = "Prix Ajusté (échelle log)", 
     ylab = "Log(Prix)", xlab = "Temps", col = "darkgreen", lwd = 1.5)
grid()

# Rendements (différence première du log)
returns <- diff(log(train_ts))
plot(returns, main = "Rendements Journaliers", 
     ylab = "Rendement", xlab = "Temps", col = "purple", lwd = 1)
abline(h = 0, col = "red", lty = 2)
grid()

# ============================================================================
# 3. STATISTIQUES DESCRIPTIVES
# ============================================================================

cat("\n========== STATISTIQUES DESCRIPTIVES ==========\n")
cat("\nSérie complète (train):\n")
print(summary(train_ts))
cat("\nÉcart-type:", sd(train_ts), "\n")
cat("Coefficient de variation:", sd(train_ts)/mean(train_ts), "\n")

cat("\nRendements journaliers:\n")
print(summary(returns))
cat("Écart-type:", sd(returns, na.rm = TRUE), "\n")

# ============================================================================
# 4. ACF et PACF - ANALYSE DE L'AUTOCORRÉLATION
# ============================================================================

par(mfrow = c(3, 2), mar = c(4, 4, 3, 2))

# ACF/PACF de la série originale
acf(train_ts, lag.max = 50, main = "ACF - Série Originale")
pacf(train_ts, lag.max = 50, main = "PACF - Série Originale")

# ACF/PACF des différences premières
diff_ts <- diff(train_ts)
acf(diff_ts, lag.max = 50, main = "ACF - Différences Premières")
pacf(diff_ts, lag.max = 50, main = "PACF - Différences Premières")

# ACF/PACF des rendements
acf(returns, lag.max = 50, main = "ACF - Rendements", na.action = na.pass)
pacf(returns, lag.max = 50, main = "PACF - Rendements", na.action = na.pass)

cat("\n========== INTERPRÉTATION ACF/PACF ==========\n")
cat("\n1. Série Originale:\n")
cat("   - ACF décroit très lentement → NON-STATIONNAIRE\n")
cat("   - Présence de forte autocorrélation → Trend évident\n")

cat("\n2. Différences Premières:\n")
cat("   - ACF chute rapidement après lag 1\n")
cat("   - PACF montre quelques pics significatifs\n")
cat("   - Suggère un processus ARIMA(p,1,q)\n")

cat("\n3. Rendements:\n")
cat("   - ACF/PACF proches de zéro → Quasi white noise\n")
cat("   - Quelques pics significatifs possibles (volatility clustering)\n")

# ============================================================================
# 5. DÉCOMPOSITION DE LA SÉRIE TEMPORELLE
# ============================================================================

par(mfrow = c(1, 1))

# Décomposition STL (Seasonal-Trend decomposition using Loess)
decomp_stl <- stl(train_ts, s.window = "periodic")
plot(decomp_stl, main = "Décomposition STL - NVIDIA")

cat("\n========== DÉCOMPOSITION STL ==========\n")
cat("\nComposantes extraites:\n")
cat("1. TREND: Tendance haussière forte et persistante\n")
cat("2. SEASONAL: Saisonnalité annuelle faible (marché financier)\n")
cat("3. REMAINDER: Résidus avec volatilité variable\n")

# Force de la tendance et saisonnalité
trend_strength <- 1 - var(decomp_stl$time.series[,"remainder"]) / 
  var(decomp_stl$time.series[,"trend"] + 
        decomp_stl$time.series[,"remainder"])
seasonal_strength <- 1 - var(decomp_stl$time.series[,"remainder"]) / 
  var(decomp_stl$time.series[,"seasonal"] + 
        decomp_stl$time.series[,"remainder"])

cat("\nForce de la tendance:", round(trend_strength, 3), 
    ifelse(trend_strength > 0.6, "(FORTE)", "(FAIBLE)"), "\n")
cat("Force de la saisonnalité:", round(seasonal_strength, 3), 
    ifelse(seasonal_strength > 0.6, "(FORTE)", "(FAIBLE)"), "\n")

# ============================================================================
# 6. DÉTECTION DES VALEURS ABERRANTES (OUTLIERS)
# ============================================================================

par(mfrow = c(2, 2), mar = c(4, 4, 3, 2))

# Méthode 1: IQR sur les rendements
Q1 <- quantile(returns, 0.25, na.rm = TRUE)
Q3 <- quantile(returns, 0.75, na.rm = TRUE)
IQR_val <- Q3 - Q1
lower_bound <- Q1 - 3 * IQR_val
upper_bound <- Q3 + 3 * IQR_val

outliers_iqr <- which(returns < lower_bound | returns > upper_bound)

plot(returns, main = "Outliers - Méthode IQR (3×IQR)", 
     ylab = "Rendements", col = "gray60")
points(outliers_iqr, returns[outliers_iqr], col = "red", pch = 19, cex = 1.5)
abline(h = c(lower_bound, upper_bound), col = "red", lty = 2)
legend("topright", legend = c("Normal", "Outliers"), 
       col = c("gray60", "red"), pch = c(1, 19))

# Méthode 2: Z-score
z_scores <- scale(returns)
outliers_z <- which(abs(z_scores) > 3)

plot(returns, main = "Outliers - Z-score (|z| > 3)", 
     ylab = "Rendements", col = "gray60")
points(outliers_z, returns[outliers_z], col = "blue", pch = 19, cex = 1.5)
abline(h = c(-3*sd(returns, na.rm=TRUE), 3*sd(returns, na.rm=TRUE)), 
       col = "blue", lty = 2)

# Méthode 3: Boxplot
boxplot(returns, main = "Boxplot des Rendements", 
        ylab = "Rendements", col = "lightblue")
points(rep(1, length(outliers_iqr)), returns[outliers_iqr], 
       col = "red", pch = 19, cex = 1.5)

# Méthode 4: Résidus de la décomposition STL
residuals_stl <- decomp_stl$time.series[,"remainder"]
outliers_stl <- tsoutliers(train_ts)

plot(residuals_stl, main = "Résidus STL - Outliers", 
     ylab = "Résidus", col = "gray60")
if(length(outliers_stl$index) > 0) {
  points(outliers_stl$index, residuals_stl[outliers_stl$index], 
         col = "darkgreen", pch = 19, cex = 1.5)
}

# ============================================================================
# 7. RAPPORT DES OUTLIERS
# ============================================================================

cat("\n========== DÉTECTION DES VALEURS ABERRANTES ==========\n")

cat("\nMéthode IQR (3×IQR):\n")
cat("  Nombre d'outliers:", length(outliers_iqr), "\n")
cat("  Pourcentage:", round(100*length(outliers_iqr)/length(returns), 2), "%\n")
if(length(outliers_iqr) > 0) {
  cat("  Dates principales:\n")
  print(head(train_data$Date[outliers_iqr + 1], 10))
  cat("  Valeurs extrêmes (rendements):\n")
  print(head(sort(returns[outliers_iqr]), 5))
  cat("  ...\n")
  print(head(sort(returns[outliers_iqr], decreasing = TRUE), 5))
}

cat("\nMéthode Z-score (|z| > 3):\n")
cat("  Nombre d'outliers:", length(outliers_z), "\n")
cat("  Pourcentage:", round(100*length(outliers_z)/length(returns), 2), "%\n")

cat("\nMéthode STL (tsoutliers):\n")
cat("  Nombre d'outliers:", length(outliers_stl$index), "\n")


# Load required packages fresh
library(forecast)
library(tseries)


# Step 1: Check if differencing is needed (determine d and D)
plot(train_ts)
adf.test(train_ts)  # If p-value > 0.05, data is non-stationary
#p value=0.9306 data us stationart



# Use ndiffs to automatically determine d
ndiffs(train_ts, test = "adf")   # Returns optimal d
ndiffs(train_ts, test = "kpss")

# For seasonal differencing (D)
nsdiffs(train_ts, test = "seas")  # Returns optimal D
nsdiffs(train_ts, test = "ocsb")

# Apply differencing
train_diff <- diff(train_ts, differences = 1)  # Regular differencing (d=1)
plot(train_diff)


# Step 2: Look at ACF and PACF to determine p, q, P, Q
par(mfrow=c(2,1))
acf(train_diff, lag.max = 251, main = "ACF - 1 Year of Lags")
pacf(train_diff, lag.max = 502, main = "ACF - 2 Year of Lags")

p=1

# Pour regressors externes (Volume)
xreg <- as.matrix(train_data["Volume"])  # Matrice des régresseurs pour train
xreg1 <- as.matrix(test_data["Volume"])  # Matrice des régresseurs pour test

# Ajustement de l'ARIMA avec régresseurs externes
library(forecast)
fit_arimax <- auto.arima(train_ts, xreg = xreg)
summary(fit_arimax)

# Prédiction sur test avec régresseurs externes
forecast_arimax <- forecast(fit_arimax, xreg = xreg1, h = length(test_ts))
plot(forecast_arimax)
lines(test_ts, col = "red")

# SARIMA
sarima_mod <- Arima(train_ts,xreg = xreg, order=c(1,1,1), seasonal=c(0,1,0))
summary(sarima_mod)
checkresiduals(sarima_mod)

# VAR (multivarié)
library(vars)

#Scale the data 
multi_data <- data.frame(
  Adj.Close = train_data$Adj.Close,
  Volume = train_data$Volume
)


# Scale to same magnitude
multi_data_scaled <- scale(multi_data)

# Convert to time series
multi_ts <- ts(multi_data_scaled, 
               start = c(as.numeric(format(train_data$Date[1], "%Y")),
                         as.numeric(format(train_data$Date[1], "%j"))),
               frequency = 251)

# Find optimal lag
lag_select <- VARselect(multi_ts, lag.max = 5, type = "both")
cat("\nOptimal lag selection:\n")
print(lag_select$selection)

# Use optimal lag (or fall back to 1 if it fails)
optimal_p <- lag_select$selection["AIC(n)"]
if(optimal_p > 3) optimal_p <- 1  # Safety check

# Fit VAR model
var_mod <- VAR(multi_ts, p = optimal_p, type = "both")
summary(var_mod)
  
# Forecast
var_forecast <- predict(var_mod, n.ahead = nrow(test_data))
plot(var_forecast)

# ARCH/GARCH (volatilité)
library(rugarch)

# GARCH models work on returns, not prices
train_returns <- diff(log(train_data$Adj.Close))  # Log returns
train_returns <- train_returns[!is.na(train_returns)]  # Remove NA

# Specify GARCH model
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(1, 1), include.mean = TRUE),
  distribution.model = "norm"
)

# Fit GARCH model
garch_mod <- ugarchfit(spec, train_returns)
show(garch_mod)

# Forecast with GARCH
garch_forecast <- ugarchforecast(garch_mod, n.ahead = nrow(test_data))
plot(garch_forecast)








