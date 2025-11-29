#!/usr/bin/Rscript


meu_estilo <- function() {
    list(
        theme_bw(base_size = 22),
        theme(
            legend.title = element_blank(),
            plot.margin = unit(c(0, 0, 0, 0), "cm"),
            legend.spacing = unit(1, "mm"),
            legend.position = "right",
            legend.justification = "left",
            legend.box.spacing = unit(0, "pt"),
            legend.box.margin = margin(0, 0, 0, 0),
            axis.text.x = element_text(angle=45, vjust=1, hjust=1)
        ))
}


options(crayon.enabled=FALSE)
suppressMessages(library(fs))
suppressMessages(library(tidyverse))
BASE <- "../"
tibble(CSV = dir_ls(BASE, regexp = "csv$", recurse=TRUE)) |>
    mutate(DATA = map(CSV, read_csv, show_col_types=FALSE, progress=FALSE)) |>
    filter(!grepl("rastro", CSV)) |>
    separate(CSV, into=paste0("XX", 1:2), sep="/") |>
    separate(XX2, into=paste0("YY", 1:2), sep="-") |>
    select(host = YY1, app = YY2, DATA) |>
    mutate(app = gsub("\\.csv", "", app)) |>
    unnest(DATA) |>
    group_by(host, app, type, problemsize, subproblemsize, threads, tasks) -> df

df |>
    summarize(comp.mean = mean(comptime),
              comp.sd = sd(comptime),
              comp.se = 3*comp.sd/sqrt(n()),
              app.mean = mean(apptime),
              app.sd = sd(apptime),
              app.se = 3*app.sd/sqrt(n()),
              .groups="keep") -> df.stats

df.stats |>
    ## Let's disregard application time.
    select(-contains("app.")) -> df.stats.base
df.stats.base |>
    ## Select the "empty" as the reference
    filter(type == "empty") |>
    ungroup(type) |> select(-type) |>
    rename_with(~ paste0("empty.", .), comp.mean:comp.se) -> df.stats.empty

df.stats.base |>
    ## Remove the "empty", our reference
    filter(type != "empty") |>
    ## Associate with the reference
    ungroup(type, tasks) |>
    left_join(df.stats.empty, by = join_by(host, app, problemsize, subproblemsize, threads)) |>
    ## Compute distances (the overhead)
    mutate(diff.mean = comp.mean - empty.comp.mean,
           diff.se = sqrt(comp.se^2 + empty.comp.se^2)) -> df.stats.diff

df.stats.diff |>
    filter(app == "qr") |>
    ggplot(aes(
        color = host,
        x = threads, y = diff.mean,
        ymin = diff.mean - diff.se,
        ymax = diff.mean + diff.se)) +
    geom_point(position="dodge") +
    geom_errorbar(width=.8, alpha=.7, position="dodge") +
    meu_estilo() +
    facet_grid(subproblemsize~type) +
    coord_cartesian(ylim=c(0, 0.15)) +
    xlab("Number of threads [count]") +
    ylab("Intrusion time [seconds]") +
    theme(legend.position="top") -> p

ggsave(
    filename = "qr-threads-overhead.pdf",
    plot = p,
    device = pdf,
    width=14, height=7
)
