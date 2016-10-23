library(ggplot2)
library(plyr)
library(lubridate)
library(reshape)

# coloured on single plot
df = read.delim("/tmp/f", h=F, sep=" ", col.names=c("run", "episode_len", "reward"))
df = ddply(df, .(run), mutate, n=seq_along(episode_len)) # adds n seq number distinst per run
ggplot(df, aes(n, reward)) +
  geom_point(alpha=0.1, aes(color=run)) +
  geom_smooth(aes(color=run))

# grid
df = read.delim("/tmp/f", h=F, sep=" ", col.names=c("run", "episode_len", "reward"))
df = ddply(df, .(run), mutate, n=seq_along(episode_len)) # adds n seq number distinst per run
ggplot(df, aes(n, reward)) +
  geom_point(alpha=0.5) +
  geom_smooth() + facet_grid(~run)

# density
df = read.delim("/tmp/g", h=F, sep=" ")
df$time_per_step = df$V2 / df$V3
head(df)
ggplot(df, aes(time_per_step*200)) + geom_histogram()

x = seq(0, 1.41, 0.005)
y = (1.5 - x) ** 5
plot(x, y)

x = seq(0, 0.6, 0.001)
y = 7 * (0.4 + 0.6 - x) ** 10
plot(x, y)

df = data.frame()
df$
df$
ggplot(df, aes(x, y)) + geom_point()

df = data.frame()
df$
df$
head(df)
ggplot(df, aes(x, y)) + geom_point()

df = read.delim("/home/mat/dev/cartpole++/rewards.action", h=F)
ggplot(df, aes(V1)) + geom_density()

df = read.delim("/tmp/f", h=F, sep="\t", col.names=c("dts", "eval"))
df$dts = ymd_hms(df$dts)
ggplot(df, aes(dts, eval)) + geom_point(alpha=0.2) + geom_smooth()
  
df = read.delim("/tmp/q", h=F, col.names=c("R", "angles", "actions"))
df = df[c("angles", "actions")]
df$both = df$angles + df$actions
df$n = seq(1:nrow(df))
df <- melt(df, id=c("n"))
ggplot(df, aes(n, value)) + geom_point(aes(color=variable))

df = read.delim("/tmp/outs", h=F, sep=" ", col.names=c("run", "n", "r"))
summary(df)
ggplot(df, aes(n, r)) +
  geom_point(alpha=0.1, aes(color=run)) +
  geom_smooth(aes(color=run)) +
  facet_grid(~run)