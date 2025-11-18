# Charger vos données
data <- read.csv("NVDA.csv")
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

# Method 1: Scale the data (RECOMMENDED)
multi_data <- data.frame(
  Adj.Close = train_data$Adj.Close,
  Volume = train_data$Volume
)

# Check original scales
cat("Price range:", range(multi_data$Adj.Close), "\n")
cat("Volume range:", range(multi_data$Volume), "\n")

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

# ===== Model Comparison =====
cat("\n===== Model Comparison =====\n")
cat("ARIMAX AIC:", fit_arimax$aic, "\n")
cat("SARIMA AIC:", sarima_mod$aic, "\n")

# Plot all forecasts together
plot(test_ts, col = "black", lwd = 2, main = "Model Comparison", 
     ylab = "Adj.Close", xlab = "Time")
lines(forecast_arimax$mean, col = "blue", lwd = 1.5)
lines(forecast_sarima$mean, col = "red", lwd = 1.5)
legend("topleft", 
       legend = c("Actual", "ARIMAX", "SARIMA"), 
       col = c("black", "blue", "red"), 
       lty = 1, lwd = c(2, 1.5, 1.5))



# ===== LOAD REQUIRED PACKAGES =====
library(keras)
install_keras()
library(tensorflow)
install_tensorflow()
library(forecast)

# ===== PREPARE DATA FOR NEURAL NETWORKS =====
cat("\n===== Preparing Data for Neural Networks =====\n")

# Normalize data (critical for neural networks)
min_price <- min(train_data$Adj.Close)
max_price <- max(train_data$Adj.Close)

normalize <- function(x) {
  (x - min_price) / (max_price - min_price)
}

denormalize <- function(x) {
  x * (max_price - min_price) + min_price
}

# Normalize train and test
train_normalized <- normalize(train_data$Adj.Close)
test_normalized <- normalize(test_data$Adj.Close)

# Create sequences for LSTM/GRU (use past n days to predict next day)
create_sequences <- function(data, n_lag = 30) {
  X <- list()
  y <- c()
  
  for(i in (n_lag + 1):length(data)) {
    X[[length(X) + 1]] <- data[(i - n_lag):(i - 1)]
    y <- c(y, data[i])
  }
  
  # Convert to array
  X <- array(unlist(X), dim = c(length(X), n_lag, 1))
  
  return(list(X = X, y = y))
}

# Create sequences (use 30 days to predict next day)
n_lag <- 30
train_seq <- create_sequences(train_normalized, n_lag)
test_seq <- create_sequences(c(train_normalized, test_normalized), n_lag)

# Split test sequences
n_train <- length(train_normalized)
test_start_idx <- n_train - n_lag + 1
test_indices <- test_start_idx:(test_start_idx + length(test_normalized) - 1)

X_train <- train_seq$X
y_train <- train_seq$y

# For testing, use sequences that start from the end of training
test_seq_full <- create_sequences(c(train_normalized, test_normalized), n_lag)
X_test <- test_seq_full$X[test_indices, , , drop = FALSE]
y_test <- test_seq_full$y[test_indices]

cat("Training samples:", dim(X_train)[1], "\n")
cat("Test samples:", dim(X_test)[1], "\n")

# ===== MODEL 1: LSTM =====
cat("\n===== Building LSTM Model =====\n")

# Clear any existing models
k_clear_session()

# Build LSTM architecture
lstm_model <- keras_model_sequential() %>%
  layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(n_lag, 1)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = 50, return_sequences = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 25) %>%
  layer_dense(units = 1)

# Compile model
lstm_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = 'mse',
  metrics = c('mae')
)

# Summary
summary(lstm_model)

# Train LSTM
cat("Training LSTM...\n")
lstm_history <- lstm_model %>% fit(
  x = X_train,
  y = y_train,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.1,
  verbose = 1,
  callbacks = list(
    callback_early_stopping(patience = 10, restore_best_weights = TRUE)
  )
)

# Plot training history
plot(lstm_history)

# Predict
lstm_pred_normalized <- lstm_model %>% predict(X_test)
lstm_pred <- denormalize(lstm_pred_normalized)

# Calculate metrics
lstm_rmse <- sqrt(mean((test_data$Adj.Close[1:length(lstm_pred)] - lstm_pred)^2))
lstm_mae <- mean(abs(test_data$Adj.Close[1:length(lstm_pred)] - lstm_pred))
cat("LSTM RMSE:", lstm_rmse, "\n")
cat("LSTM MAE:", lstm_mae, "\n")


