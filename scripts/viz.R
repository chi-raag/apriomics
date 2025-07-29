library(tidyverse)
library(ggplot2)
library(ggpattern)


data <- read_csv("./output/benchmark_prior_recovery_results.csv")

data_filtered <- data %>%
    filter(Method != "LLM-Informed Hierarchical") %>%
    mutate(
        llm = case_when(
            str_detect(Method, "Flash 2.0") ~ "Flash 2.0",
            str_detect(Method, "GPT-4.1") ~ "GPT-4.1",
            str_detect(Method, "O4-Mini") ~ "O4-Mini",
            str_detect(Method, "Uninformative") ~ "Uninformative",
            str_detect(Method, "Oracle") ~ "Oracle"
        ),
        context = case_when(
            str_detect(Method, "With Context") ~ "Context",
            str_detect(Method, "No Context") ~ "No Context",
            TRUE ~ NA
        ),
        short_ctx = case_when(
            context == "Context" ~ "C",
            context == "No Context" ~ "N"
        )
    )

sample_labeller <- labeller(
    sample_size = c(
        `5`  = "n = 5",
        `10` = "n = 10",
        `15` = "n = 15",
        `20` = "n = 20"
    )
)

label_data <- data_filtered %>%
    filter(!is.na(context)) %>%
    group_by(sample_size, Method, context, short_ctx) %>%
    summarize(max_rmse = max(rmse, na.rm = TRUE), .groups = "drop") %>%
    group_by(sample_size) %>%
    mutate(y_pos = max(max_rmse) * 1) %>%
    ungroup()

data_filtered %>%
    ggplot(aes(x = Method, y = rmse, fill = llm)) +
    geom_boxplot(outliers = FALSE) +
    geom_text(
        inherit.aes = FALSE,
        data = label_data,
        aes(x = Method, y = y_pos, label = short_ctx, color = context),
        size = 3,
        show.legend = TRUE
    ) +
    scale_color_manual(
        name = "Context",
        values = c(
            "Context" = "black",
            "No Context" = "black"
        ),
        breaks = c("Context", "No Context"),
        labels = c("With Context", "No Context"),
        guide = guide_legend(
            override.aes = list(label = c("C", "N"), size = 4)
        )
    ) +
    facet_grid(~sample_size, scales = "free_x", labeller = sample_labeller) +
    theme_bw() +
    theme(
        axis.text.x  = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.x = element_blank()
    ) +
    labs(
        y     = "Root Mean Squared Error (RMSE)",
        fill  = "LLM Method",
        color = "Context"
    )
