# Comprehensive Clustering Analysis: K-Means, K-Modes, and K-Prototypes (Enhanced)
# This script combines and compares three clustering techniques with improved logging and visualization.

# --- 0. Load Necessary Packages ---
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, cluster, factoextra, ggplot2, klaR, reshape2, plotly, tidyr, patchwork)

# Helper function for categorical modes
get_mode_info <- function(data_col) {
  tab <- table(data_col)
  mode_val <- names(tab)[which.max(tab)]
  mode_freq <- max(tab) / sum(tab)
  return(paste0(mode_val, " (", round(mode_freq * 100, 1), "%)"))
}

# --- 1. Load and Prepare Data ---
cat("\n[1/6] Loading and merging datasets...\n")
user_continuous_data <- read.csv("user_behavior_data.csv")
user_categorical_data <- read.csv("user_categorical_data.csv", stringsAsFactors = TRUE)

merged_user_data <- full_join(user_continuous_data, 
                              user_categorical_data %>% dplyr::select(-contains("Cluster")), 
                              by = "UserID")
cat("Data loaded successfully. Total users:", nrow(merged_user_data), "\n")

# --- 2. K-Means Clustering (Numerical Data) ---
cat("\n[2/6] Running K-Means Clustering (Numerical Only)...\n")
kmeans_data <- user_continuous_data %>% dplyr::select(-UserID) %>% scale()

cat("Iterating K (1-10) for Elbow and Silhouette methods...\n")
kmeans_metrics <- data.frame(K = 1:10, WCSS = NA, Silhouette = NA)
for (k in 1:10) {
  set.seed(42)
  km_tmp <- kmeans(kmeans_data, centers = k, nstart = 25)
  kmeans_metrics$WCSS[k] <- km_tmp$tot.withinss
  
  if (k > 1) {
    sil_tmp <- silhouette(km_tmp$cluster, dist(kmeans_data))
    kmeans_metrics$Silhouette[k] <- mean(sil_tmp[, 3])
  }
  cat(sprintf("K = %2d | WCSS: %10.2f | Silhouette: %s\n", 
              k, kmeans_metrics$WCSS[k], ifelse(k==1, "N/A", round(kmeans_metrics$Silhouette[k], 4))))
}

# 2.1 K-Means Elbow & Silhouette Plots
p_km_elbow <- ggplot(kmeans_metrics, aes(x = K, y = WCSS)) +
  geom_line(color = "blue") + geom_point(color = "red") +
  geom_vline(xintercept = 4, linetype = "dashed", color = "darkgreen") +
  annotate("text", x = 4.2, y = max(kmeans_metrics$WCSS) * 0.9, label = "Optimal K=4", color = "darkgreen") +
  labs(title = "K-Means: Elbow Method", y = "Total WCSS") + theme_minimal()
print(ggplotly(p_km_elbow))

p_km_sil <- ggplot(kmeans_metrics[-1,], aes(x = K, y = Silhouette)) +
  geom_line(color = "blue") + geom_point(color = "red") +
  geom_vline(xintercept = 4, linetype = "dashed", color = "darkgreen") +
  annotate("text", x = 4.2, y = max(kmeans_metrics[-1,]$Silhouette) * 0.9, label = "Optimal K=4", color = "darkgreen") +
  labs(title = "K-Means: Silhouette Method", y = "Avg Silhouette Width") + theme_minimal()
print(ggplotly(p_km_sil))

# Apply K-Means
optimal_k_means <- 4
set.seed(42)
kmeans_model <- kmeans(kmeans_data, centers = optimal_k_means, nstart = 25)
user_continuous_data$KMeans_Cluster <- as.factor(kmeans_model$cluster)

# 2.2 PCA Visualization
# Using fviz_cluster for robust ellipse generation
p1_cluster_pca <- fviz_cluster(kmeans_model, data = kmeans_data,
                               geom = "point",
                               stand = FALSE,
                               ellipse.type = "norm",
                               palette = "Set1",
                               main = "K-Means: PCA Cluster Visualization") +
  theme_minimal()
print(ggplotly(p1_cluster_pca))

# 2.3 Feature Scatter Plot
p_km_scatter <- ggplot(user_continuous_data, aes(x = TimeSpentMinutes, y = PastPurchases, color = KMeans_Cluster)) +
  geom_point(alpha = 0.6) +
  labs(title = "K-Means: Time Spent vs. Past Purchases", x = "Time Spent (Min)", y = "Past Purchases") +
  theme_minimal() + scale_color_brewer(palette = "Set1")
print(ggplotly(p_km_scatter))

# --- 3. K-Modes Clustering (Categorical Data) ---
cat("\n[3/6] Running K-Modes Clustering (Categorical Only)...\n")
kmodes_data <- user_categorical_data %>% dplyr::select(-UserID) %>% mutate_all(as.factor)

cat("Iterating K (1-10) to find optimal K (Cost/Dissimilarity)...\n")
kmodes_metrics <- data.frame(K = 1:10, Cost = NA)
for (k in 1:10) {
  set.seed(42)
  km_model_tmp <- klaR::kmodes(kmodes_data, k, iter.max = 50)
  kmodes_metrics$Cost[k] <- sum(km_model_tmp$withindiff)
  cat(sprintf("K = %2d | Total Dissimilarity (Cost): %10.2f\n", k, kmodes_metrics$Cost[k]))
}

# 3.1 K-Modes Cost Plot
p_kmodes_cost <- ggplot(kmodes_metrics, aes(x = K, y = Cost)) +
  geom_line(color = "blue") + geom_point(color = "red") +
  geom_vline(xintercept = 4, linetype = "dashed", color = "darkgreen") +
  annotate("text", x = 4.2, y = max(kmodes_metrics$Cost) * 0.9, label = "Suggested K=4", color = "darkgreen") +
  labs(title = "K-Modes: Cost vs. Number of Clusters", x = "K", y = "Total Dissimilarity") + theme_minimal()
print(ggplotly(p_kmodes_cost))

# Apply K-Modes
optimal_k_modes <- 4
set.seed(42)
kmodes_model <- klaR::kmodes(kmodes_data, optimal_k_modes, iter.max = 100)
user_categorical_data$KModes_Cluster <- as.factor(kmodes_model$cluster)

# 3.2 K-Modes Distribution Plots
p_kmodes_device <- ggplot(user_categorical_data, aes(x = DeviceType, fill = KModes_Cluster)) +
  geom_bar(position = "fill") + labs(title = "K-Modes: Device Type Distribution", y = "Proportion") + 
  theme_minimal() + scale_fill_brewer(palette = "Set2")
print(ggplotly(p_kmodes_device))

p_kmodes_content <- ggplot(user_categorical_data, aes(x = PreferredContentCategory, fill = KModes_Cluster)) +
  geom_bar(position = "fill") + labs(title = "K-Modes: Content Category Distribution", y = "Proportion") + 
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + scale_fill_brewer(palette = "Set2")
print(ggplotly(p_kmodes_content))

# --- 4. K-Prototypes / PAM Clustering (Mixed Data) ---
cat("\n[4/6] Running PAM Clustering (Mixed Numerical & Categorical)...\n")
clustering_data_mixed <- merged_user_data %>% dplyr::select(-UserID)
categorical_cols <- c("DeviceType", "PreferredContentCategory", "SubscriptionPlan", "UserOrigin", "FeedbackGiven")
clustering_data_mixed <- clustering_data_mixed %>% mutate_at(vars(all_of(categorical_cols)), as.factor)

# Calculate Gower Distance Matrix
gower_dist <- daisy(clustering_data_mixed, metric = "gower")

cat("Iterating K (1-10) for PAM (K-Prototypes equivalent)...\n")
pam_metrics <- data.frame(K = 1:10, Cost = NA, Silhouette = NA)
for (k in 1:10) {
  set.seed(42)
  pam_temp <- pam(gower_dist, k = k, diss = TRUE, stand = FALSE)
  pam_metrics$Cost[k] <- pam_temp$objective["swap"] 
  
  if(k > 1) {
    pam_metrics$Silhouette[k] <- pam_temp$silinfo$avg.width
  }
  
  cat(sprintf("K = %2d | Total Dissimilarity: %10.4f | Avg Silhouette: %s\n", 
              k, pam_metrics$Cost[k], ifelse(k==1, "N/A", round(pam_metrics$Silhouette[k], 4))))
}

# 4.1 PAM Elbow & Silhouette Plots
p_pam_elbow <- ggplot(pam_metrics, aes(x = K, y = Cost)) +
  geom_line(color = "blue") + geom_point(color = "red") +
  geom_vline(xintercept = 4, linetype = "dashed", color = "darkgreen") +
  annotate("text", x = 4.2, y = max(pam_metrics$Cost) * 0.9, label = "Suggested K=4", color = "darkgreen") +
  labs(title="PAM: Elbow Method (Total Dissimilarity)") + theme_minimal()
print(ggplotly(p_pam_elbow))

p_pam_sil <- ggplot(pam_metrics[-1,], aes(x = K, y = Silhouette)) +
  geom_line(color = "blue") + geom_point(color = "red") +
  geom_vline(xintercept = 4, linetype = "dashed", color = "darkgreen") +
  annotate("text", x = 4.2, y = max(pam_metrics[-1,]$Silhouette) * 0.9, label = "Suggested K=4", color = "darkgreen") +
  labs(title="PAM: Silhouette Method") + theme_minimal()
print(ggplotly(p_pam_sil))

# Apply PAM
optimal_k_proto <- 4
set.seed(42)
pam_model <- pam(gower_dist, k = optimal_k_proto, diss = TRUE, stand = FALSE, nstart = 25)
merged_user_data$PAM_Cluster <- as.factor(pam_model$clustering)

# 4.2 PAM Scatter and Distribution Plots
p_pam_scatter <- ggplot(merged_user_data, aes(x = TimeSpentMinutes, y = PastPurchases, color = PAM_Cluster)) +
  geom_point(alpha = 0.6) + labs(title = "PAM Clusters: Time Spent vs. Past Purchases") + 
  theme_minimal() + scale_color_brewer(palette = "Set1")
print(ggplotly(p_pam_scatter))

p_pam_device <- ggplot(merged_user_data, aes(x = DeviceType, fill = PAM_Cluster)) +
  geom_bar(position = "fill") + labs(title = "PAM Clusters: Device Type Distribution") + 
  theme_minimal() + scale_fill_brewer(palette = "Set2")
print(ggplotly(p_pam_device))

# 4.3 PAM PCA Visualization (Numerical dimensions)
# Extract only numeric columns for PCA projection and ensure it's a data frame
pam_numeric_data <- as.data.frame(clustering_data_mixed %>% dplyr::select(where(is.numeric)) %>% scale())

# Using fviz_cluster without ellipses for clearer point distribution
p_pam_pca <- fviz_cluster(list(data = pam_numeric_data, cluster = pam_model$clustering),
                          geom = "point",
                          stand = FALSE,
                          ellipse = FALSE,
                          palette = "Set1",
                          main = "PAM: PCA Cluster Visualization (Numerical Features)") +
  theme_minimal()
print(ggplotly(p_pam_pca))

# 4.4 3D PAM Visualization
cat("\nGenerating 3D PAM Visualization...\n")
p5_3d <- plot_ly(merged_user_data, 
                 x = ~ProductsViewed, 
                 y = ~TimeSpentMinutes, 
                 z = ~PastPurchases, 
                 color = ~PAM_Cluster, 
                 colors = "Set1",
                 text = ~paste("UserID:", UserID,
                               "<br>Device:", DeviceType,
                               "<br>Plan:", SubscriptionPlan,
                               "<br>Origin:", UserOrigin,
                               "<br>Feedback:", FeedbackGiven),
                 hoverinfo = "text+x+y+z",
                 type = "scatter3d", 
                 mode = "markers", 
                 marker = list(size = 4)) %>%
  layout(title = list(text = "3D PAM Clusters (Numerical Axes)"),
         scene = list(xaxis = list(title = 'Products Viewed'),
                      yaxis = list(title = 'Time Spent'),
                      zaxis = list(title = 'Past Purchases')))
print(p5_3d)

# --- 5. Comparative Analysis ---
cat("\n[5/6] Generating Comparative Metrics...\n")

# Consolidate results
comparison_df <- merged_user_data %>%
  left_join(user_continuous_data %>% dplyr::select(UserID, KMeans_Cluster), by = "UserID") %>%
  left_join(user_categorical_data %>% dplyr::select(UserID, KModes_Cluster), by = "UserID")

# Print Summary Tables to Console
cat("\n--- K-Means (Numerical) Cluster Means ---\n")
print(comparison_df %>% group_by(KMeans_Cluster) %>% summarize(across(where(is.numeric) & -UserID, mean)))

cat("\n--- K-Modes (Categorical) Cluster Modes ---\n")
print(comparison_df %>% group_by(KModes_Cluster) %>% summarize(across(all_of(categorical_cols), get_mode_info)))

cat("\n--- PAM (Mixed) Cluster Summary ---\n")
print(comparison_df %>% group_by(PAM_Cluster) %>% 
        summarize(Avg_Time = mean(TimeSpentMinutes), 
                  Mode_Device = get_mode_info(DeviceType),
                  N = n()))

# Heatmap: K-Means vs PAM Overlap
p4_heatmap_data <- as.data.frame(table(comparison_df$KMeans_Cluster, comparison_df$PAM_Cluster))
colnames(p4_heatmap_data) <- c("KMeans", "PAM", "Count")

p4_heat_plot <- ggplot(p4_heatmap_data, aes(x = KMeans, y = PAM, fill = Count)) +
  geom_tile(color = "white", size = 0.2) +
  geom_text(aes(label = Count), fontface = "bold") +
  scale_fill_distiller(palette = "YlOrRd", direction = 1) +
  labs(title = "Cluster Overlap: K-Means vs. PAM", 
       subtitle = "Intensity shows how well clusters from different methods align",
       x = "K-Means Cluster ID", y = "PAM Cluster ID") +
  theme_minimal()
print(ggplotly(p4_heat_plot))

# --- 6. Final Export ---
cat("\n[6/6] Finalizing Analysis...\n")
write.csv(comparison_df, "consolidated_clustering_results.csv", row.names = FALSE)
cat("\nAnalysis complete. Results printed to console and saved to CSV.\n")
