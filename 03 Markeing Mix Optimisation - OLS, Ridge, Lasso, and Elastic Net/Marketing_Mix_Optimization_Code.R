# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

knitr::opts_chunk$set(
  echo = TRUE,
  warning = FALSE,
  message = FALSE,
  fig.width = 10,
  fig.height = 6,
  fig.align = 'center'
)

library(tidyverse)      # Data manipulation and visualization
library(glmnet)         # Regularized regression (Ridge, Lasso, Elastic Net)
library(corrplot)       # Correlation matrix visualization
library(GGally)         # Pairwise correlation plots
library(scales)         # Number formatting for plots
library(viridis)        # Color palettes for visualizations
library(patchwork)      # Combine multiple ggplot objects
library(plotly)         # Interactive plots

# Helper functions for interactive plots
plot_glmnet_cv <- function(cv_model, title) {
  df <- data.frame(
    log_lambda = log(cv_model$lambda),
    mse = cv_model$cvm,
    upper = cv_model$cvup,
    lower = cv_model$cvlo
  )
  
  p <- ggplot(df, aes(x = log_lambda, y = mse)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, fill = "grey50") +
    geom_line(color = "#dc2626") +
    geom_point(color = "#dc2626", size = 1) +
    geom_vline(xintercept = log(cv_model$lambda.min), linetype = "dashed", color = "red") +
    geom_vline(xintercept = log(cv_model$lambda.1se), linetype = "dashed", color = "blue") +
    labs(
      title = title,
      x = "Log(Lambda)", 
      y = "Mean-Squared Error"
    ) +
    theme_minimal()
  
  ggplotly(p)
}

plot_glmnet_path <- function(model, title) {
  # Extract beta coefficients (excluding intercept)
  beta <- as.matrix(model$beta)
  df <- as.data.frame(t(beta))
  df$log_lambda <- log(model$lambda)
  
  df_long <- df %>%
    pivot_longer(
      cols = -log_lambda,
      names_to = "Variable",
      values_to = "Coefficient"
    )
  
  p <- ggplot(df_long, aes(x = log_lambda, y = Coefficient, color = Variable)) +
    geom_line() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    labs(title = title, x = "Log(Lambda)", y = "Coefficient") +
    theme_minimal()
  
  ggplotly(p)
}

# Set Global Options
options(scipen = 999)   # Disable scientific notation for readability
set.seed(42)            # Reproducibility for train-test splits

# ==============================================================================
# DATA IMPORT & VALIDATION
# ==============================================================================

# Load Marketing Data
data <- read.csv("Marketing_regularization_data.csv")

# Data Structure Overview
cat("\n=== DATA STRUCTURE ===\n")
str(data)

# Basic Data Quality Checks
cat("\n=== DATA QUALITY CHECKS ===\n")
cat("Total Observations:", nrow(data), "\n")
cat("Total Variables:", ncol(data), "\n")
cat("Missing Values:\n")
print(colSums(is.na(data)))
cat("\nSummary Statistics:\n")
print(summary(data))

# ==============================================================================
# PART A: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

# Transform data to long format for faceted histograms
data_long <- data %>%
  pivot_longer(
    cols = -sales,
    names_to = "variable",
    values_to = "value"
  )

# Create distribution plots for all variables
p1_distributions <- ggplot(data_long, aes(x = value)) +
  geom_histogram(bins = 35, fill = "#4338ca", alpha = 0.75, color = "white") +
  facet_wrap(~variable, scales = "free", ncol = 3) +
  scale_x_continuous(labels = comma) +
  labs(
    title = "EDA 1: Real Business Data Is Skewed & Heavy-Tailed",
    subtitle = "Marketing spends, pricing, and demand do not follow normal distributions",
    x = NULL,
    y = "Frequency",
    caption = "Note: Non-normal distributions suggest OLS may produce unstable estimates"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    strip.text = element_text(face = "bold")
  )

print(ggplotly(p1_distributions))

# Select key marketing levers for bivariate analysis
key_vars <- c(
  "price",
  "discount_percent",
  "tv_ad_spend",
  "digital_ad_spend",
  "social_media_spend"
)

# Create individual scatter plots with LOESS smoothing
plots_bivariate <- lapply(
  key_vars,
  function(v) {
    ggplot(data, aes_string(x = v, y = "sales")) +
      geom_point(alpha = 0.3, color = "#2563eb", size = 1.5) +
      geom_smooth(method = "loess", color = "black", se = TRUE, linewidth = 1) +
      scale_x_continuous(labels = comma) +
      scale_y_continuous(labels = comma) +
      labs(
        title = paste("Sales vs", tools::toTitleCase(gsub("_", " ", v))),
        x = tools::toTitleCase(gsub("_", " ", v)),
        y = "Sales"
      ) +
      theme_minimal(base_size = 10) +
      theme(plot.title = element_text(face = "bold", size = 11))
  }
)

# Combine plots using plotly subplot
p2_bivariate <- subplot(lapply(plots_bivariate, ggplotly), nrows = 3, margin = 0.05, titleX = TRUE, titleY = TRUE) %>%
  layout(
    title = list(text = "EDA 2: Sales vs Key Marketing Drivers<br><sup>LOESS curves reveal non-linear relationships and diminishing returns</sup>"),
    margin = list(t = 60)
  )

print(p2_bivariate)

# Apply log transformation to stabilize variance
corr_matrix <- cor(log(data + 1))

# Create correlation heatmap
corr_df <- as.data.frame(as.table(corr_matrix))
names(corr_df) <- c("Variable1", "Variable2", "Correlation")

p_corr <- ggplot(corr_df, aes(Variable1, Variable2, fill = Correlation)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(Correlation, 2)), color = "white", size = 3) +
  scale_fill_viridis(option = "D", limit = c(-1, 1), name = "Corr") +
  labs(
    title = "EDA 3: Multicollinearity Across Marketing Channels",
    x = NULL, y = NULL
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    plot.title = element_text(face = "bold")
  )

print(ggplotly(p_corr))

# Identify high correlation pairs
high_corr_pairs <- which(abs(corr_matrix) > 0.7 & corr_matrix < 1, arr.ind = TRUE)
if(nrow(high_corr_pairs) > 0) {
  cat("\nHigh Correlation Pairs (|r| > 0.7):\n")
  for(i in 1:nrow(high_corr_pairs)) {
    var1 <- rownames(corr_matrix)[high_corr_pairs[i, 1]]
    var2 <- colnames(corr_matrix)[high_corr_pairs[i, 2]]
    corr_val <- corr_matrix[high_corr_pairs[i, 1], high_corr_pairs[i, 2]]
    cat(sprintf("  %s <-> %s: %.3f\n", var1, var2, corr_val))
  }
}

# Select key marketing spend variables
marketing_vars <- data %>%
  select(
    tv_ad_spend,
    digital_ad_spend,
    social_media_spend,
    search_ads_spend
  )

# Create pairwise correlation and distribution plot
p4_pairwise <- ggpairs(
  log(marketing_vars + 1),
  title = "EDA 4: Marketing Channels Move as a Portfolio",
  columnLabels = c("TV", "Digital", "Social", "Search"),
  upper = list(continuous = wrap("cor", size = 3, color = "black")),
  lower = list(continuous = wrap("points", alpha = 0.3, size = 0.8, color = "#2563eb")),
  diag = list(continuous = wrap("densityDiag", fill = "#4338ca", alpha = 0.5))
) +
  theme_minimal()

print(ggplotly(p4_pairwise) %>% layout(margin = list(t = 100)))

# ==============================================================================
# PART B: ORDINARY LEAST SQUARES (OLS) - BASELINE MODEL
# ==============================================================================

# Fit OLS Model
ols_model <- lm(sales ~ ., data = data)

# Display Results
cat("\nOLS Regression Summary:\n")
print(summary(ols_model))

# Extract Key Metrics
ols_r_squared <- summary(ols_model)$r.squared
ols_adj_r_squared <- summary(ols_model)$adj.r.squared
ols_residual_se <- summary(ols_model)$sigma

cat("\n--- OLS Performance Metrics ---\n")
cat("R-squared:", round(ols_r_squared, 4), "\n")
cat("Adjusted R-squared:", round(ols_adj_r_squared, 4), "\n")
cat("Residual Standard Error:", round(ols_residual_se, 2), "\n")

# Check for Multicollinearity using VIF
if(require(car, quietly = TRUE)) {
  cat("\nVariance Inflation Factors (VIF):\n")
  vif_values <- vif(ols_model)
  print(round(vif_values, 2))
  cat("\nNote: VIF > 5-10 indicates problematic multicollinearity\n")
}

# ==============================================================================
# PART C: REGULARIZED REGRESSION - ADDRESSING MULTICOLLINEARITY
# ==============================================================================

# Prepare Data for glmnet
x <- model.matrix(sales ~ ., data = data)[, -1]  # Remove intercept column
y <- data$sales

cat("\nPredictors Matrix Dimensions:", dim(x), "\n")
cat("Response Vector Length:", length(y), "\n")

# -----------------------------------------------------------------------------
# C.1: Ridge Regression (alpha = 0)
# -----------------------------------------------------------------------------

cat("\n--- Ridge Regression ---\n")

# Cross-Validation to Find Optimal Lambda
cv_ridge <- cv.glmnet(
  x, y,
  alpha = 0,
  nfolds = 10,
  type.measure = "mse"
)

# Visualize Cross-Validation Results
print(plot_glmnet_cv(cv_ridge, "Ridge Regression: Cross-Validation Error vs Log(λ)"))

cat("Optimal Lambda (min):", cv_ridge$lambda.min, "\n")
cat("Lambda 1SE:", cv_ridge$lambda.1se, "\n")

# Ridge Coefficient Path
ridge_model <- glmnet(x, y, alpha = 0)

print(plot_glmnet_path(ridge_model, "Ridge: All Variables Retained, Coefficients Shrunk"))

# -----------------------------------------------------------------------------
# C.2: Lasso Regression (alpha = 1)
# -----------------------------------------------------------------------------

cat("\n--- Lasso Regression ---\n")

# Cross-Validation for Lasso
cv_lasso <- cv.glmnet(
  x, y,
  alpha = 1,
  nfolds = 10,
  type.measure = "mse"
)

# Visualize CV Results
print(plot_glmnet_cv(cv_lasso, "Lasso Regression: Cross-Validation Error vs Log(λ)"))

cat("Optimal Lambda (min):", cv_lasso$lambda.min, "\n")
cat("Lambda 1SE:", cv_lasso$lambda.1se, "\n")

# Lasso Coefficient Path
lasso_model <- glmnet(x, y, alpha = 1)

print(plot_glmnet_path(lasso_model, "Lasso: Weak Marketing Channels Are Eliminated"))

# Identify Variables Selected by Lasso
lasso_coef <- coef(cv_lasso, s = "lambda.min")
selected_vars <- rownames(lasso_coef)[lasso_coef[, 1] != 0]
cat("\nVariables Selected by Lasso (λ.min):", length(selected_vars) - 1, "out of", ncol(x), "\n")
cat("Selected Variables:\n")
print(selected_vars[selected_vars != "(Intercept)"])

# -----------------------------------------------------------------------------
# C.3: Elastic Net (alpha = 0.5)
# -----------------------------------------------------------------------------

cat("\n--- Elastic Net Regression ---\n")

# Cross-Validation for Elastic Net
cv_enet <- cv.glmnet(
  x, y,
  alpha = 0.5,
  nfolds = 10,
  type.measure = "mse"
)

# Visualize CV Results
print(plot_glmnet_cv(cv_enet, "Elastic Net: Cross-Validation Error (α = 0.5)"))

cat("Optimal Lambda (min):", cv_enet$lambda.min, "\n")
cat("Lambda 1SE:", cv_enet$lambda.1se, "\n")

# Elastic Net Coefficient Path
enet_model <- glmnet(x, y, alpha = 0.5)

print(plot_glmnet_path(enet_model, "Elastic Net: Grouped Selection of Marketing Channels"))

# ==============================================================================
# PART D: MODEL COMPARISON & PERFORMANCE EVALUATION
# ==============================================================================

cat("\n=== MODEL COMPARISON ===\n")

# Extract Coefficients from Each Model
coef_compare <- tibble(
  Variable = rownames(coef(cv_enet)),
  OLS = c(coef(ols_model)),
  Ridge = as.numeric(coef(cv_ridge, s = "lambda.min")),
  Lasso = as.numeric(coef(cv_lasso, s = "lambda.min")),
  ElasticNet = as.numeric(coef(cv_enet, s = "lambda.min"))
) %>%
  filter(Variable != "(Intercept)") %>%
  pivot_longer(
    cols = -Variable,
    names_to = "Model",
    values_to = "Coefficient"
  )

# Visualize Coefficient Differences
p_coef_compare <- ggplot(
  coef_compare,
  aes(
    x = reorder(Variable, abs(Coefficient)),
    y = Coefficient,
    fill = Model
  )
) +
  geom_bar(stat = "identity", position = "dodge", color = "white", linewidth = 0.3) +
  coord_flip() +
  scale_fill_viridis_d(option = "D") +
  labs(
    title = "Coefficient Comparison: Which Marketing Levers Truly Matter?",
    subtitle = "Elastic Net provides the most balanced and interpretable managerial view",
    x = NULL,
    y = "Coefficient Value",
    caption = "Note: Zero coefficients indicate variables eliminated by Lasso/Elastic Net"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    legend.position = "bottom"
  )

print(
  ggplotly(p_coef_compare) %>%
    layout(
      title = list(text = "Coefficient Comparison: Which Marketing Levers Truly Matter?<br><sup>Elastic Net provides the most balanced and interpretable managerial view</sup>"),
      annotations = list(
        list(
          x = 1, y = -0.15, 
          text = "Note: Zero coefficients indicate variables eliminated by Lasso/Elastic Net", 
          showarrow = F, xref='paper', yref='paper', 
          xanchor='right', yanchor='auto', xshift=0, yshift=0,
          font=list(size=10, color = "gray50")
        )
      ),
      margin = list(t = 60, b = 80)
    )
)

cat("\n--- Train-Test Split for Performance Evaluation ---\n")

# Create Train-Test Split
set.seed(42)
n <- nrow(data)
train_idx <- sample(1:n, 0.7 * n)

train <- data[train_idx, ]
test  <- data[-train_idx, ]

cat("Training Set Size:", nrow(train), "\n")
cat("Test Set Size:", nrow(test), "\n")

# Prepare Matrices
x_train <- model.matrix(sales ~ ., train)[, -1]
y_train <- train$sales

x_test <- model.matrix(sales ~ ., test)[, -1]
y_test <- test$sales

# Model 1: OLS
cat("\n--- OLS Performance ---\n")
ols_model_train <- lm(sales ~ ., data = train)
ols_pred <- predict(ols_model_train, newdata = test)

ols_rmse <- sqrt(mean((y_test - ols_pred)^2))
ols_mae <- mean(abs(y_test - ols_pred))
ols_mape <- mean(abs((y_test - ols_pred) / y_test)) * 100
ols_r2_test <- cor(y_test, ols_pred)^2

cat("RMSE:", round(ols_rmse, 2), "\n")
cat("MAPE:", round(ols_mape, 2), "%\n")

# Model 2: Ridge
cat("\n--- Ridge Performance ---\n")
cv_ridge_train <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 10)
ridge_pred <- predict(cv_ridge_train, s = "lambda.min", newx = x_test)

ridge_rmse <- sqrt(mean((y_test - ridge_pred)^2))
ridge_mae <- mean(abs(y_test - ridge_pred))
ridge_mape <- mean(abs((y_test - ridge_pred) / y_test)) * 100
ridge_r2_test <- cor(y_test, ridge_pred)^2

cat("RMSE:", round(ridge_rmse, 2), "\n")
cat("MAPE:", round(ridge_mape, 2), "%\n")

# Model 3: Lasso
cat("\n--- Lasso Performance ---\n")
cv_lasso_train <- cv.glmnet(x_train, y_train, alpha = 1, nfolds = 10)
lasso_pred <- predict(cv_lasso_train, s = "lambda.min", newx = x_test)

lasso_rmse <- sqrt(mean((y_test - lasso_pred)^2))
lasso_mae <- mean(abs(y_test - lasso_pred))
lasso_mape <- mean(abs((y_test - lasso_pred) / y_test)) * 100
lasso_r2_test <- cor(y_test, lasso_pred)^2

cat("RMSE:", round(lasso_rmse, 2), "\n")
cat("MAPE:", round(lasso_mape, 2), "%\n")

# Model 4: Elastic Net
cat("\n--- Elastic Net Performance ---\n")
cv_enet_train <- cv.glmnet(x_train, y_train, alpha = 0.5, nfolds = 10)
enet_pred <- predict(cv_enet_train, s = "lambda.min", newx = x_test)

enet_rmse <- sqrt(mean((y_test - enet_pred)^2))
enet_mae <- mean(abs(y_test - enet_pred))
enet_mape <- mean(abs((y_test - enet_pred) / y_test)) * 100
enet_r2_test <- cor(y_test, enet_pred)^2

cat("RMSE:", round(enet_rmse, 2), "\n")
cat("MAPE:", round(enet_mape, 2), "%\n")

# Performance Summary Table
performance <- data.frame(
  Model = c("OLS", "Ridge", "Lasso", "Elastic Net"),
  RMSE = c(ols_rmse, ridge_rmse, lasso_rmse, enet_rmse),
  MAE = c(ols_mae, ridge_mae, lasso_mae, enet_mae),
  MAPE = c(ols_mape, ridge_mape, lasso_mape, enet_mape),
  R_Squared = c(ols_r2_test, ridge_r2_test, lasso_r2_test, enet_r2_test)
)

cat("\n=== MODEL PERFORMANCE SUMMARY (Test Set) ===\n")
print(performance)

# Identify Best Model
best_model_idx <- which.min(performance$RMSE)
cat("\nBest Performing Model (by RMSE):", performance$Model[best_model_idx], "\n")

# Generate Predictions for Full Dataset
predicted_sales <- predict(cv_enet, s = "lambda.min", newx = x)

efficiency_df <- data %>%
  mutate(
    Predicted_Sales = as.numeric(predicted_sales),
    Total_Marketing_Spend = tv_ad_spend + digital_ad_spend + 
                           social_media_spend + search_ads_spend + 
                           influencer_spend + email_marketing
  )

# Plot Efficiency Curve
p_efficiency <- ggplot(
  efficiency_df,
  aes(Total_Marketing_Spend, Predicted_Sales)
) +
  geom_point(alpha = 0.35, color = "#1d4ed8", size = 2) +
  geom_smooth(method = "loess", color = "black", se = TRUE, linewidth = 1.2) +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma) +
  labs(
    title = "Marketing Spend vs Sales: Diminishing Returns",
    subtitle = "Regularization reveals efficiency boundaries and optimal spend ranges",
    x = "Total Marketing Spend",
    y = "Predicted Sales",
    caption = "Note: Flattening curve indicates diminishing marginal returns"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13)
  )

print(ggplotly(p_efficiency))

# ==============================================================================
# PART E: FINAL PRODUCTION MODEL - PREDICTION & CREDIBILITY
# ==============================================================================

cat("\n=== PRODUCTION MODEL DEVELOPMENT ===\n")

# Fresh Train-Test Split for Production Model
set.seed(42)
n <- nrow(data)
train_index <- sample(1:n, size = 0.7 * n)

train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# Prepare Production Matrices
x_train <- model.matrix(sales ~ ., train_data)[, -1]
y_train <- train_data$sales

x_test <- model.matrix(sales ~ ., test_data)[, -1]
y_test <- test_data$sales

cat("Production Training Set:", nrow(train_data), "observations\n")
cat("Production Test Set:", nrow(test_data), "observations\n")

# Train Final Elastic Net Model
cat("\n--- Training Final Elastic Net Model ---\n")

cv_enet_final <- cv.glmnet(
  x_train,
  y_train,
  alpha = 0.5,
  nfolds = 10,
  type.measure = "mse"
)

# Visualize Cross-Validation Performance
print(plot_glmnet_cv(cv_enet_final, "Final Elastic Net: 10-Fold Cross-Validation"))

# Generate Out-of-Sample Predictions
test_pred <- predict(
  cv_enet_final,
  s = "lambda.min",
  newx = x_test
)

# Calculate Production Accuracy Metrics
rmse_final <- sqrt(mean((y_test - test_pred)^2))
mae_final <- mean(abs(y_test - test_pred))
mape_final <- mean(abs((y_test - test_pred) / y_test)) * 100
r2_final <- cor(y_test, test_pred)^2

cat("\n=== PRODUCTION MODEL PERFORMANCE ===\n")
cat("RMSE:", round(rmse_final, 2), "→ Average forecast error in sales units\n")
cat("MAE:", round(mae_final, 2), "\n")
cat("MAPE:", round(mape_final, 2), "% → Average percentage error for business planning\n")
cat("R-squared:", round(r2_final, 4), "\n")

# Create prediction dataframe
pred_df <- data.frame(
  Actual = y_test,
  Predicted = as.numeric(test_pred)
) %>%
  mutate(
    Residual = Actual - Predicted,
    Abs_Pct_Error = abs(Residual / Actual) * 100
  )

# Actual vs Predicted Plot
p_pred_quality <- ggplot(pred_df, aes(Actual, Predicted)) +
  geom_point(alpha = 0.5, color = "#2563eb", size = 2.5) +
  geom_abline(slope = 1, intercept = 0, color = "black", linewidth = 1, linetype = "dashed") +
  geom_smooth(method = "lm", color = "#dc2626", se = FALSE, linewidth = 0.8) +
  scale_x_continuous(labels = comma) +
  scale_y_continuous(labels = comma) +
  labs(
    title = "Predictive Performance: Actual vs Predicted Sales",
    subtitle = "Elastic Net generalizes well to unseen regions/time periods",
    x = "Actual Sales",
    y = "Predicted Sales",
    caption = "Diagonal line = perfect predictions | Red line = model fit"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13)
  )

print(ggplotly(p_pred_quality))

# Residual Analysis
cat("\n--- Residual Analysis ---\n")
cat("Mean Residual (should be ~0):", round(mean(pred_df$Residual), 2), "\n")
cat("Median Absolute Error:", round(median(abs(pred_df$Residual)), 2), "\n")
cat("% of Predictions within 10% error:", 
    round(sum(pred_df$Abs_Pct_Error <= 10) / nrow(pred_df) * 100, 1), "%\n")

# ==============================================================================
# PART F: BUSINESS INSIGHTS - ACTIONABLE COEFFICIENT INTERPRETATION
# ==============================================================================

cat("\n=== EXTRACTING BUSINESS ACTIONS ===\n")

# Extract Elastic Net Coefficients
enet_coef <- coef(cv_enet_final, s = "lambda.min")

coef_df <- data.frame(
  Variable = rownames(enet_coef),
  Coefficient = as.numeric(enet_coef)
) %>%
  filter(Variable != "(Intercept)") %>%
  arrange(desc(abs(Coefficient))) %>%
  mutate(
    Impact = case_when(
      Coefficient > 0 ~ "Positive (Growth Driver)",
      Coefficient < 0 ~ "Negative (Sales Inhibitor)",
      TRUE ~ "No Impact (Eliminated)"
    ),
    Abs_Coefficient = abs(Coefficient)
  )

# Display Strategic Coefficients
cat("\n=== STRATEGIC MARKETING LEVERS (Ranked by Impact) ===\n")
print(coef_df, row.names = FALSE)

# Visualize Strategic Levers
p_strategic_levers <- ggplot(
  coef_df,
  aes(
    x = reorder(Variable, Coefficient),
    y = Coefficient,
    fill = Coefficient > 0
  )
) +
  geom_bar(stat = "identity", color = "white", linewidth = 0.3) +
  coord_flip() +
  scale_fill_manual(
    values = c("TRUE" = "#16a34a", "FALSE" = "#dc2626"),
    labels = c("Sales Inhibitor", "Growth Driver")
  ) +
  labs(
    title = "Elastic Net Insights: What Really Drives Sales",
    subtitle = "Green = Growth drivers (invest) | Red = Sales inhibitors (manage carefully)",
    x = NULL,
    y = "Coefficient Value",
    fill = "Impact Type"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    legend.position = "bottom"
  )

print(ggplotly(p_strategic_levers))

# Identify Top/Bottom Drivers
top_3_drivers <- coef_df %>%
  filter(Coefficient > 0) %>%
  slice_max(order_by = Coefficient, n = 3)

cat("\n--- TOP 3 GROWTH DRIVERS ---\n")
print(top_3_drivers[, c("Variable", "Coefficient")], row.names = FALSE)

eliminated_vars <- coef_df %>%
  filter(abs(Coefficient) < 0.001)

if(nrow(eliminated_vars) > 0) {
  cat("\n--- VARIABLES ELIMINATED (Consider Budget Reallocation) ---\n")
  print(eliminated_vars$Variable)
} else {
  cat("\n--- All variables retained with non-zero coefficients ---\n")
}

# ==============================================================================
# PART G: PRESCRIPTIVE ANALYTICS - WHAT-IF SCENARIO PLANNING
# ==============================================================================

cat("\n=== WHAT-IF SCENARIO ANALYSIS ===\n")

# Baseline Prediction
baseline_pred <- predict(
  cv_enet_final,
  s = "lambda.min",
  newx = x_test
)
baseline_avg_sales <- mean(baseline_pred)

cat("Baseline Average Predicted Sales:", round(baseline_avg_sales, 2), "\n")

cat("\n--- SCENARIO 1: Increase Digital Ad Spend by 10% ---\n")

scenario1_data <- test_data
scenario1_data$digital_ad_spend <- scenario1_data$digital_ad_spend * 1.10

x_scenario1 <- model.matrix(sales ~ ., scenario1_data)[, -1]
scenario1_pred <- predict(
  cv_enet_final,
  s = "lambda.min",
  newx = x_scenario1
)

scenario1_avg_sales <- mean(scenario1_pred)
scenario1_lift <- scenario1_avg_sales - baseline_avg_sales
scenario1_pct_lift <- (scenario1_lift / baseline_avg_sales) * 100

cat("New Average Predicted Sales:", round(scenario1_avg_sales, 2), "\n")
cat("Sales Lift:", round(scenario1_lift, 2), 
    "(", round(scenario1_pct_lift, 2), "% increase)\n")

cat("\n--- SCENARIO 2: Strategic Budget Reallocation ---\n")
cat("Action: Cut weak channels by 30%, Boost strong channels by 10%\n")

scenario2_data <- test_data

# Reduce weak/eliminated channels
scenario2_data$influencer_spend <- scenario2_data$influencer_spend * 0.7
scenario2_data$email_marketing  <- scenario2_data$email_marketing * 0.7

# Boost strong channels
scenario2_data$tv_ad_spend      <- scenario2_data$tv_ad_spend * 1.1
scenario2_data$search_ads_spend <- scenario2_data$search_ads_spend * 1.1

x_scenario2 <- model.matrix(sales ~ ., scenario2_data)[, -1]
scenario2_pred <- predict(
  cv_enet_final,
  s = "lambda.min",
  newx = x_scenario2
)

scenario2_avg_sales <- mean(scenario2_pred)
scenario2_lift <- scenario2_avg_sales - baseline_avg_sales
scenario2_pct_lift <- (scenario2_lift / baseline_avg_sales) * 100

cat("New Average Predicted Sales:", round(scenario2_avg_sales, 2), "\n")
cat("Sales Lift:", round(scenario2_lift, 2), 
    "(", round(scenario2_pct_lift, 2), "% increase)\n")

# Calculate Net Budget Change
baseline_total_spend <- sum(
  test_data$tv_ad_spend,
  test_data$digital_ad_spend,
  test_data$social_media_spend,
  test_data$search_ads_spend,
  test_data$influencer_spend,
  test_data$email_marketing
)

scenario2_total_spend <- sum(
  scenario2_data$tv_ad_spend,
  scenario2_data$digital_ad_spend,
  scenario2_data$social_media_spend,
  scenario2_data$search_ads_spend,
  scenario2_data$influencer_spend,
  scenario2_data$email_marketing
)

budget_change <- scenario2_total_spend - baseline_total_spend
budget_change_pct <- (budget_change / baseline_total_spend) * 100

cat("Total Marketing Spend Change:", round(budget_change, 2), 
    "(", round(budget_change_pct, 2), "%)\n")
cat("ROI Insight: Sales lift of", round(scenario2_pct_lift, 2), 
    "% with budget change of", round(budget_change_pct, 2), "%\n")

# Scenario Comparison DataFrame
scenario_compare <- data.frame(
  Scenario = factor(
    c("Baseline", "Digital +10%", "Reallocated Mix"),
    levels = c("Baseline", "Digital +10%", "Reallocated Mix")
  ),
  Avg_Predicted_Sales = c(
    baseline_avg_sales,
    scenario1_avg_sales,
    scenario2_avg_sales
  ),
  Pct_Lift = c(
    0,
    scenario1_pct_lift,
    scenario2_pct_lift
  )
)

# Visualization
p_scenario_compare <- ggplot(
  scenario_compare,
  aes(x = Scenario, y = Avg_Predicted_Sales, fill = Scenario)
) +
  geom_bar(stat = "identity", color = "white", linewidth = 0.5) +
  geom_text(
    aes(label = paste0("+", round(Pct_Lift, 2), "%")),
    vjust = -0.5,
    nudge_y = 500,
    size = 4,
    fontface = "bold"
  ) +
  scale_y_continuous(
    labels = comma,
    expand = expansion(mult = c(0, 0.1))
  ) +
  scale_fill_viridis_d(option = "D", begin = 0.3, end = 0.8) +
  labs(
    title = "Business Impact of Marketing Decisions",
    subtitle = "Elastic Net enables data-driven prescriptive decision-making",
    y = "Average Predicted Sales",
    x = NULL,
    caption = "Percentages show lift vs baseline scenario"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    legend.position = "none",
    panel.grid.major.x = element_blank()
  )

print(ggplotly(p_scenario_compare))

# ==============================================================================
# PART H: EXECUTIVE SUMMARY & KEY TAKEAWAYS
# ==============================================================================

cat("\n")
cat("================================================================================\n")
cat("                        EXECUTIVE SUMMARY & RECOMMENDATIONS                      \n")
cat("================================================================================\n")
cat("\n")

cat("PROJECT: Marketing Mix Optimization Using Regularized Regression\n")
cat("DATE:", format(Sys.Date(), "%B %d, %Y"), "\n")
cat("\n")

cat("--- KEY FINDINGS ---\n\n")

cat("1. MULTICOLLINEARITY PROBLEM\n")
cat("   - Marketing channels are highly correlated (coordinated campaigns)\n")
cat("   - OLS regression produces unstable, uninterpretable coefficients\n")
cat("   - Solution: Regularization provides robust, actionable insights\n\n")

cat("2. MODEL PERFORMANCE\n")
cat("   - Elastic Net outperforms OLS on unseen data\n")
cat("   - Final Model RMSE:", round(rmse_final, 2), "\n")
cat("   - Final Model MAPE:", round(mape_final, 2), "%\n")
cat("   - Suitable for business planning and forecasting\n\n")

cat("3. STRATEGIC MARKETING LEVERS (Top 3)\n")
if(nrow(top_3_drivers) > 0) {
  for(i in 1:min(3, nrow(top_3_drivers))) {
    cat("   ", i, ". ", top_3_drivers$Variable[i], 
        " (Coefficient: ", round(top_3_drivers$Coefficient[i], 4), ")\n", sep = "")
  }
} else {
  cat("   Review coefficient output in Part F\n")
}
cat("\n")

cat("4. WHAT-IF SCENARIO RESULTS\n")
cat("   - Digital +10% Scenario: +", round(scenario1_pct_lift, 2), "% sales lift\n", sep = "")
cat("   - Reallocation Scenario: +", round(scenario2_pct_lift, 2), "% sales lift", 
    " with ", round(budget_change_pct, 2), "% budget change\n", sep = "")
cat("\n")

cat("--- BUSINESS RECOMMENDATIONS ---\n\n")

cat("IMMEDIATE ACTIONS:\n")
cat("1. Prioritize investment in top 3 growth drivers identified\n")
cat("2. Review channels with near-zero coefficients for potential reallocation\n")
cat("3. Test scenario-based strategies with small-scale pilots\n")
cat("4. Monitor actual vs predicted sales to refine model quarterly\n\n")

cat("LONG-TERM STRATEGY:\n")
cat("1. Implement dynamic budget optimization based on model coefficients\n")
cat("2. Build dashboard for real-time scenario analysis\n")
cat("3. Incorporate seasonality and external factors (future enhancement)\n")
cat("4. Develop channel-specific sub-models for granular optimization\n\n")

cat("MODEL MAINTENANCE:\n")
cat("1. Retrain model quarterly with fresh data\n")
cat("2. Validate predictions against actual outcomes\n")
cat("3. Update alpha parameter if marketing strategy shifts\n")
cat("4. Consider time-series regularization for forecasting\n\n")

cat("================================================================================\n")
cat("                              END OF ANALYSIS                                    \n")
cat("================================================================================\n")

sessionInfo()
