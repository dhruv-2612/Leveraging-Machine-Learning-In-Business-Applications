# ============================================================================
# CUSTOMER CHURN PREDICTION — CLASSIFICATION CASE STUDY
# ============================================================================
# Purpose: Build logistic regression model to predict customer churn
# Output: Risk-prioritized customer list for retention campaigns
# ============================================================================


# ============================================================================
# 0. ENVIRONMENT SETUP
# ============================================================================

# Load Required Libraries
library(dplyr)          # Data manipulation
library(plotly)         # Interactive visualizations
library(caret)          # Machine learning framework
library(tidyr)          # Data tidying
library(reshape2)       # Data reshaping for plots
library(pROC)           # ROC curve and AUC metrics
library(car)            # VIF for multicollinearity check
library(randomForest)   # Random forest comparison model


# ============================================================================
# 1. DATA IMPORT & PREPARATION
# ============================================================================

# Load Customer Churn Dataset
churn_data <- read.csv(
  "customer_churn_mba_case_FINAL.csv",
  stringsAsFactors = FALSE
)

# Convert Variables to Appropriate Types
# Categorical variables: contract type, customer segment
churn_data$contract_type     <- factor(churn_data$contract_type)
churn_data$customer_segment <- factor(churn_data$customer_segment)
# Target variable: churn (0=No, 1=Yes)
churn_data$churn            <- factor(churn_data$churn, levels = c(0,1))

# Inspect Dataset Structure
str(churn_data)


# ----------------------------------------------------------------------------
# 1.1 Data Quality Assessment
# ----------------------------------------------------------------------------

cat("Dataset dimensions:", dim(churn_data), "\n")
summary(churn_data)
cat("Missing values per column:\n")
print(sapply(churn_data, function(x) sum(is.na(x))))
cat("Duplicate rows:", sum(duplicated(churn_data)), "\n")


# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

# ----------------------------------------------------------------------------
# 2.1 Overall Churn Rate (Target Variable Distribution)
# ----------------------------------------------------------------------------

churn_rate <- churn_data %>%
  count(churn) %>%
  mutate(percent = n / sum(n) * 100)

# Visualize: Class balance check (important for model evaluation)
plot_ly(
  churn_rate,
  x = ~churn,
  y = ~percent,
  color = ~churn,
  colors = c("0"="#28a745", "1"="#dc3545"),
  type = "bar",
  text = ~paste0(round(percent,1), "%"),
  textposition = "outside",
  marker = list(line = list(color = "white", width = 1))
) %>%
  layout(
    title = "Overall Customer Churn Rate",
    xaxis = list(title = "Churn Status (0=No, 1=Yes)"),
    yaxis = list(title = "Percentage (%)")
  )


# ----------------------------------------------------------------------------
# 2.2 Distribution of Key Numerical Variables
# ----------------------------------------------------------------------------
# Check for skewness, outliers, and typical value ranges

num_vars <- churn_data %>%
  select(tenure_months, monthly_usage, avg_monthly_bill, engagement_score)

num_vars_long <- melt(num_vars)

plot_ly(
  num_vars_long,
  x = ~value,
  color = ~variable,
  colors = c("tenure_months"="#1f77b4", "monthly_usage"="#ff7f0e", "avg_monthly_bill"="#2ca02c", "engagement_score"="#d62728"),
  type = "histogram",
  opacity = 0.6,
  nbinsx = 30
) %>%
  layout(
    title = "Distribution of Key Numerical Metrics",
    barmode = "overlay",
    xaxis = list(title = "Metric Value"),
    yaxis = list(title = "Frequency")
  )


# ============================================================================
# 3. BIVARIATE ANALYSIS — WHO IS CHURNING?
# ============================================================================
# Analyze churn rates across different customer segments to identify patterns

# ----------------------------------------------------------------------------
# 3.1 Churn Rate by Contract Type
# ----------------------------------------------------------------------------
# Hypothesis: Monthly contracts may have higher churn than annual/long-term

contract_churn <- churn_data %>%
  group_by(contract_type) %>%
  summarise(
    churn_rate = mean(as.numeric(as.character(churn))),
    .groups = "drop"
  )

plot_ly(
  contract_churn,
  x = ~contract_type,
  y = ~churn_rate,
  color = ~contract_type,
  colors = "Dark2",
  type = "bar",
  text = ~paste0(round(churn_rate*100,1), "%"),
  textposition = "outside",
  marker = list(line = list(color = "white", width = 1))
) %>%
  layout(
    title = "Churn Rate by Contract Type",
    xaxis = list(title = "Contract Type"),
    yaxis = list(title = "Churn Rate (%)")
  )


# ----------------------------------------------------------------------------
# 3.2 Churn Rate by Customer Segment
# ----------------------------------------------------------------------------
# Identify which segments (e.g., Enterprise, SMB, Consumer) are at risk

segment_churn <- churn_data %>%
  group_by(customer_segment) %>%
  summarise(
    churn_rate = mean(as.numeric(as.character(churn))),
    .groups = "drop"
  )

plot_ly(
  segment_churn,
  x = ~customer_segment,
  y = ~churn_rate,
  color = ~customer_segment,
  colors = "Set1",
  type = "bar",
  text = ~paste0(round(churn_rate*100,1), "%"),
  textposition = "outside",
  marker = list(line = list(color = "white", width = 1))
) %>%
  layout(
    title = "Churn Rate by Customer Segment",
    xaxis = list(title = "Customer Segment"),
    yaxis = list(title = "Churn Rate (%)")
  )


# ----------------------------------------------------------------------------
# 3.3 Impact of Price Hikes & Payment Delays on Churn
# ----------------------------------------------------------------------------
# Investigate behavioral triggers: price sensitivity and payment friction

binary_vars <- churn_data %>%
  group_by(price_hike_flag, payment_delay_flag) %>%
  summarise(
    churn_rate = mean(as.numeric(as.character(churn))),
    .groups = "drop"
  )

plot_ly(
  binary_vars,
  x = ~price_hike_flag,
  y = ~churn_rate,
  color = ~as.factor(payment_delay_flag),
  colors = c("0"="#87CEEB", "1"="#4682B4"),
  type = "bar",
  text = ~paste0(round(churn_rate*100,1), "%"),
  textposition = "outside",
  marker = list(line = list(color = "white", width = 1))
) %>%
  layout(
    title = "Churn Rate: Price Hikes & Payment Delays",
    xaxis = list(title = "Price Hike Flag (0=No, 1=Yes)"),
    yaxis = list(title = "Churn Rate (%)")
  )


# ============================================================================
# 4. TESTING INTUITIONS — BOXPLOT COMPARISONS
# ============================================================================
# Challenge assumptions: Do engaged/high-usage customers really stay?

# ----------------------------------------------------------------------------
# 4.1 Monthly Usage vs Churn
# ----------------------------------------------------------------------------
# Does higher usage correlate with lower churn? (Not always!)

plot_ly(
  churn_data,
  x = ~churn,
  y = ~monthly_usage,
  type = "box",
  color = ~churn,
  colors = c("0"="#28a745", "1"="#dc3545")
) %>%
  layout(
    title = "Monthly Usage Distribution by Churn",
    xaxis = list(title = "Churn (0=No, 1=Yes)"),
    yaxis = list(title = "Monthly Usage")
  )


# ----------------------------------------------------------------------------
# 4.2 Engagement Score vs Churn
# ----------------------------------------------------------------------------
# Quantify relationship between engagement and retention

plot_ly(
  churn_data,
  x = ~churn,
  y = ~engagement_score,
  type = "box",
  color = ~churn,
  colors = c("0"="#28a745", "1"="#dc3545")
) %>%
  layout(
    title = "Engagement Score Distribution by Churn",
    xaxis = list(title = "Churn (0=No, 1=Yes)"),
    yaxis = list(title = "Engagement Score")
  )


# ============================================================================
# 5. CORRELATION ANALYSIS (EXPLORATORY)
# ============================================================================
# Identify multicollinearity risks before modeling

corr_data <- churn_data %>%
  mutate(churn = as.numeric(as.character(churn))) %>%
  select_if(is.numeric)

corr_matrix <- cor(corr_data)

# Heatmap: Red = positive correlation, Blue = negative correlation
plot_ly(
  x = colnames(corr_matrix),
  y = rownames(corr_matrix),
  z = corr_matrix,
  type = "heatmap",
  colorscale = "RdYlBu_r"
) %>%
  layout(title = "Feature Correlation Heatmap")


# ============================================================================
# 6. SCATTER PLOTS — WHY CLASSIFICATION IS NEEDED
# ============================================================================
# Visualize class overlap: No clear linear boundaries = need classification

# ----------------------------------------------------------------------------
# 6.1 Usage vs Engagement (Feature Interaction)
# ----------------------------------------------------------------------------

plot_ly(
  churn_data,
  x = ~monthly_usage,
  y = ~engagement_score,
  color = ~churn,
  colors = c("0"="#28a745", "1"="#dc3545"),
  type = "scatter",
  mode = "markers",
  marker = list(opacity = 0.6, size=4)
) %>%
  layout(
    title = "Monthly Usage vs Engagement Score",
    xaxis = list(title = "Monthly Usage"),
    yaxis = list(title = "Engagement Score")
  )


# ----------------------------------------------------------------------------
# 6.2 Tenure vs Revenue (Loyalty-Value Matrix)
# ----------------------------------------------------------------------------

plot_ly(
  churn_data,
  x = ~tenure_months,
  y = ~avg_monthly_bill,
  color = ~churn,
  colors = c("0"="#28a745", "1"="#dc3545"),
  type = "scatter",
  mode = "markers",
  marker = list(opacity = 0.6, size=4)
) %>%
  layout(
    title = "Tenure vs Average Monthly Bill",
    xaxis = list(title = "Tenure (Months)"),
    yaxis = list(title = "Avg Monthly Bill ($)")
  )


# ----------------------------------------------------------------------------
# 6.3 Service Friction Indicators (Complaints & Support)
# ----------------------------------------------------------------------------
# Bubble size = engagement; color = churn status

plot_ly(
  churn_data,
  x = ~complaints_last_6m,
  y = ~support_calls,
  color = ~churn,
  colors = c("0"="#28a745", "1"="#dc3545"),
  size = ~engagement_score,
  type = "scatter",
  mode = "markers",
  marker = list(opacity = 0.7),
  sizes = c(5, 30)
) %>%
  layout(
    title = "Complaints vs Support Calls by Churn",
    xaxis = list(title = "Complaints (Last 6 Months)"),
    yaxis = list(title = "Support Calls")
  )


# ============================================================================
# 7. MODEL DEVELOPMENT
# ============================================================================

# ----------------------------------------------------------------------------
# 7.1 Train-Test Split (70-30)
# ----------------------------------------------------------------------------
# Stratified split to preserve churn rate in both sets

set.seed(123)  # Reproducibility
train_index <- createDataPartition(churn_data$churn, p = 0.7, list = FALSE)

train_data <- churn_data[train_index, ]
test_data  <- churn_data[-train_index, ]

# Verify balance preservation
cat("Train churn rate:", mean(as.numeric(as.character(train_data$churn))), "\n")
cat("Test churn rate:", mean(as.numeric(as.character(test_data$churn))), "\n")


# ----------------------------------------------------------------------------
# 7.2 Logistic Regression Model (Baseline)
# ----------------------------------------------------------------------------
# Predicts probability of churn using all relevant features

logit_model <- glm(
  churn ~ tenure_months + monthly_usage + complaints_last_6m +
    support_calls + payment_delay_flag + avg_monthly_bill +
    discount_received + price_hike_flag +
    contract_type + customer_segment + engagement_score,
  data = train_data,
  family = "binomial"
)

# Model Summary: Coefficients, significance levels (p-values)
summary(logit_model)


# ----------------------------------------------------------------------------
# 7.3 Model Diagnostics
# ----------------------------------------------------------------------------

# Multicollinearity Check: VIF > 5-10 indicates problematic correlation
vif(logit_model)

# Odds Ratios: Interpretable effect sizes (e.g., OR=2 means 2x odds increase)
odds_ratios <- exp(coef(logit_model))
print(odds_ratios)


# ----------------------------------------------------------------------------
# 7.4 Cross-Validation (Model Stability Check)
# ----------------------------------------------------------------------------
# 5-fold CV to assess generalization performance

# Prepare binary factor for caret (requires "Yes"/"No" levels)
train_data$churn_cv <- factor(
  ifelse(train_data$churn == 1, "Yes", "No"),
  levels = c("No", "Yes")
)

cv_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,         # Enable probability predictions
  summaryFunction = twoClassSummary  # ROC, Sensitivity, Specificity
)

cv_model <- train(
  churn_cv ~ tenure_months + monthly_usage + complaints_last_6m +
    support_calls + payment_delay_flag + avg_monthly_bill +
    discount_received + price_hike_flag +
    contract_type + customer_segment + engagement_score,
  data = train_data,
  method = "glm",
  family = "binomial",
  metric = "ROC",             # Optimize for AUC
  trControl = cv_control
)

# CV Performance Metrics (ROC AUC, Sensitivity, Specificity)
cv_model$results


# ----------------------------------------------------------------------------
# 7.5 Model Comparison: Random Forest (Advanced Ensemble)
# ----------------------------------------------------------------------------
# Compare logistic regression vs tree-based model

rf_model <- train(
  churn_cv ~ tenure_months + monthly_usage + complaints_last_6m +
    support_calls + payment_delay_flag + avg_monthly_bill +
    discount_received + price_hike_flag +
    contract_type + customer_segment + engagement_score,
  data = train_data,
  method = "rf",
  metric = "ROC",
  trControl = cv_control
)

# Model Comparison Summary
cat("Logistic Regression CV ROC AUC:", max(cv_model$results$ROC), "\n")
cat("Random Forest CV ROC AUC:", max(rf_model$results$ROC), "\n")

# Feature Importance from Random Forest
print(varImp(rf_model, scale=FALSE))


# ============================================================================
# 8. MODEL EVALUATION ON TEST SET
# ============================================================================

# ----------------------------------------------------------------------------
# 8.1 Generate Predictions (Probabilities)
# ----------------------------------------------------------------------------

test_data$churn_prob <- predict(logit_model, test_data, type = "response")

# Visualize: Probability distribution by actual churn status
plot_ly(
  test_data,
  x = ~churn_prob,
  color = ~churn,
  colors = c("0"="#28a745", "1"="#dc3545"),
  type = "histogram",
  opacity = 0.6,
  nbinsx = 50
) %>%
  layout(
    title = "Distribution of Predicted Churn Probabilities (Test Set)",
    xaxis = list(title = "Predicted Churn Probability"),
    yaxis = list(title = "Frequency"),
    barmode = "overlay"
  )


# ----------------------------------------------------------------------------
# 8.2 ROC Curve & AUC (Threshold-Independent Performance)
# ----------------------------------------------------------------------------

roc_obj <- roc(as.numeric(test_data$churn) - 1, test_data$churn_prob, 
               levels = c(0, 1), direction = "<")
plot(roc_obj, main = "ROC Curve (Logistic Regression on Test Set)")
cat("Test Set ROC AUC:", auc(roc_obj), "\n")


# ----------------------------------------------------------------------------
# 8.3 Confusion Matrix (At 0.5 Threshold)
# ----------------------------------------------------------------------------
# Classification metrics: Accuracy, Sensitivity (Recall), Precision, F1

threshold <- 0.5
test_data$churn_pred <- ifelse(test_data$churn_prob > threshold, 1, 0)

# Confusion Matrix: positive="1" ensures churn is the focus class
confusionMatrix(
  factor(test_data$churn_pred, levels = c(0,1)),
  test_data$churn,
  positive = "1"
)


# ============================================================================
# 9. BUSINESS OUTPUT — CHURN RISK PRIORITIZATION
# ============================================================================
# Translate model into actionable customer intervention list

# ----------------------------------------------------------------------------
# 9.1 Score Entire Customer Base
# ----------------------------------------------------------------------------

# Predict churn probability for ALL customers (not just test set)
churn_data$churn_probability <- predict(
  logit_model,
  churn_data,
  type = "response"
)


# ----------------------------------------------------------------------------
# 9.2 Create Risk Buckets (Business-Friendly Segmentation)
# ----------------------------------------------------------------------------
# High Risk (≥70%): Immediate intervention required
# Medium Risk (40-70%): Monitor closely, proactive engagement
# Low Risk (<40%): Standard retention programs

churn_data$churn_risk_bucket <- case_when(
  churn_data$churn_probability >= 0.70 ~ "High Risk",
  churn_data$churn_probability >= 0.40 ~ "Medium Risk",
  TRUE ~ "Low Risk"
)

# Risk Bucket Performance Check
table(churn_data$churn_risk_bucket, churn_data$churn)


# ----------------------------------------------------------------------------
# 9.3 Executive Report: Prioritized Customer List
# ----------------------------------------------------------------------------

# Select actionable columns for business teams
final_churn_output <- churn_data %>%
  select(
    tenure_months,
    monthly_usage,
    avg_monthly_bill,
    contract_type,
    customer_segment,
    engagement_score,
    churn_probability,
    churn_risk_bucket
  ) %>%
  arrange(desc(churn_probability))  # Sort by risk (highest first)

# Preview top 10 highest-risk customers
head(final_churn_output, 10)

# Export to CSV for CRM/retention teams
write.csv(
  final_churn_output,
  "customer_churn_risk_prioritization.csv",
  row.names = FALSE
)


# ============================================================================
# END OF ANALYSIS
# ============================================================================
# Next Steps:
# 1. Deploy model to production (monthly/weekly scoring)
# 2. A/B test retention campaigns on high-risk customers
# 3. Monitor model performance drift over time
# 4. Iterate with new features (e.g., product usage patterns, NPS scores)
# ============================================================================
