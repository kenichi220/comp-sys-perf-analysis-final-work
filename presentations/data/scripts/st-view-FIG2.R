#!/usr/bin/Rscript

## Modification from pj_gantt from: https://github.com/schnorr/pajeng

library(tidyverse)
library(fs)
library(patchwork)

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

BASE <- "../st-results"
read_paje_trace <- function(file) {
  df <- read.csv(file, header=FALSE, strip.white=TRUE)
  names(df) <- c("Nature","ResourceId","Type","Start","End","Duration", "Depth", "Value")
  m <- min(df$Start)
  df$Start <- df$Start - m
  df$End <- df$Start+df$Duration
  df$Origin <- NULL
  df$Nature <- NULL
  df$Depth <- NULL
  df
}

tibble(CSV = dir_ls(BASE, regexp = "csv$", recurse=TRUE)) |>
  mutate(DATA = map(CSV, read_paje_trace)) |>
  separate(CSV, into=paste0("XX", 1:4), sep="/") |> print() |>
  separate(XX3, into=paste0("YY", 1:3), sep="-") |>
  mutate(group=paste(YY1, YY2, YY3, sep="-")) |>
  select(app = YY2, group, DATA) |>
  unnest(DATA)-> df

df <- df[df$Type == "TASK",];
df <- df[df$Duration != 0,];
df
df |> filter(app=="qr") -> df.qr
df |> filter(app=="gauss")  -> df.gauss

gauss <- ggplot(df.gauss, aes(xmin=Start,xmax=End, ymin=as.integer(factor(ResourceId)),ymax=as.integer(factor(ResourceId))+0.9, fill=Value)) +
  facet_wrap(~group) + theme_bw() + geom_rect()
qr <- ggplot(df.qr, aes(xmin=Start,xmax=End, ymin=as.integer(factor(ResourceId)),ymax=as.integer(factor(ResourceId))+0.9, fill=Value)) +
  facet_wrap(~group) + theme_bw() + geom_rect()

p <- gauss / qr

ggsave(
  filename = "st-view-figure2.pdf",
  plot = p,
  device = pdf,
  width=8, height=4
)
