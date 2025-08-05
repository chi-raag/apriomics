library(tidyverse)
library(ggplot2)

data <- read_csv("output/benchmark_prior_recovery_results.csv")

data_filtered <- data %>%
    filter(Method != "LLM-Informed Hierarchical") %>%
    mutate(
        llm = case_when(
            str_detect(Method, "Flash 2.0") ~ "Flash 2.0",
            str_detect(Method, "Pro") ~ "Pro 2.5",
            str_detect(Method, "GPT-4o") ~ "GPT-4o",
            str_detect(Method, "O3 Mini") ~ "O3 Mini",
            str_detect(Method, "Uninformative") ~ "Uninformative",
            str_detect(Method, "Oracle") ~ "Oracle",
            TRUE ~ "O3"
        ),
        llm = factor(llm, levels = c("Uninformative", "Flash 2.0", "GPT-4o", "O3", "O3 Mini", "Pro 2.5", "Oracle")),
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
    mutate(Method = factor(Method, levels = c(
        "Uninformative Bayesian",
        "Flash 2.0 (No Context, Conservative)", "Flash 2.0 (With Context, Conservative)",
        "GPT-4o (No Context, Conservative)", "GPT-4o (With Context, Conservative)",
        "O3 (No Context, Conservative)",
        "O3 Mini (No Context, Conservative)", "O3 Mini (With Context, Conservative)",
        "Pro 2.5 (No Context, Conservative)", "Pro 2.5 (With Context, Conservative)",
        "Oracle Bayesian"
    ))) %>%
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
    theme_bw(base_size = 20, base_family = "Avenir Next LT Pro") +
    theme(
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.x = element_blank(),
        axis.line.x = element_blank(),
        panel.grid = element_blank()
    ) +
    labs(
        y     = "Root Mean Squared Error (RMSE)",
        fill  = "LLM Method",
        color = "Context"
    )

ggsave("~/Downloads/tmptmptmptmp.png", width = 13, height = 9)

data_bias_var <- read_csv("output/benchmark_bias_variance_results.csv")

# compute arrow endpoints on the same filtered subset
arrow_coords <- data_bias_var |>
    filter(method %in% c("o3_no_context_conservative", "uninformative_bayesian", "oracle_bayesian")) |>
    summarise(
        min_bias = min(overall_bias, na.rm = TRUE),
        min_var  = min(overall_variance, na.rm = TRUE),
        max_bias = max(overall_bias, na.rm = TRUE),
        max_var  = max(overall_variance, na.rm = TRUE)
    ) |>
    mutate(
        good_x = min_bias + 0.2 * (max_bias - min_bias),
        good_y = min_var + 0.2 * (max_var - min_var),
        bad_x  = min_bias + 0.8 * (max_bias - min_bias),
        bad_y  = min_var + 0.8 * (max_var - min_var)
    )

data_bias_var |>
    filter(method %in% c("o3_no_context_conservative", "uninformative_bayesian", "oracle_bayesian")) |>
    mutate(
        method = case_when(
            method == "o3_no_context_conservative" ~ "O3 No Context",
            method == "uninformative_bayesian" ~ "Uninformative Bayesian",
            method == "oracle_bayesian" ~ "Oracle Bayesian"
        )
    ) |>
    ggplot(aes(x = overall_bias, y = overall_variance, color = method)) +
    geom_point(size = 6) +
    annotate(
        "text",
        x = -Inf, y = -Inf,
        label = "Low bias & variance",
        hjust = -0.5, vjust = -.5,
        size = 3, color = "darkgreen",
        family = "Avenir Next LT Pro"
    ) +
    annotate(
        "text",
        x = Inf, y = Inf,
        label = "High bias & variance",
        hjust = 1.5, vjust = 1.5,
        size = 3, color = "red",
        family = "Avenir Next LT Pro"
    ) +
    # arrows pointing into the Good/Bad corners
    geom_segment(
        data = arrow_coords,
        aes(x = good_x, y = good_y, xend = min_bias, yend = min_var),
        arrow = arrow(length = unit(0.2, "inches")),
        color = "darkgreen"
    ) +
    geom_segment(
        data = arrow_coords,
        aes(x = bad_x, y = bad_y, xend = max_bias, yend = max_var),
        arrow = arrow(length = unit(0.2, "inches")),
        color = "red"
    ) +
    facet_wrap(~sample_size, labeller = labeller(sample_labeller)) +
    scale_color_brewer(palette = "Dark2", name = "Method") +
    scale_shape_manual(values = c(16, 17, 15), name = "Method") +
    labs(
        x = "Overall Bias",
        y = "Overall Variance"
    ) +
    theme_bw(base_family = "Avenir Next LT Pro", base_size = 20)

ggsave("~/Downloads/bias_variance_plot.png", width = 15, height = 8)
