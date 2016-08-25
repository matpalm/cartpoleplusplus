library(ggplot2)
library(grid)
library(gridExtra)

# run parse_foo.sh first

df = read.delim("/tmp/actions_pre_noise", h=F, sep=" ", col.names=c("x", "y"))
png("/tmp/plots/01_x_y_scatter.png")
ggplot(df, aes(x, y)) + geom_point()
dev.off()
df$n = 1:nrow(df)
png("/tmp/plots/02_x_pre_noise.png")
ggplot(df, aes(n, x)) + geom_point() + labs(title="x pre noise")
dev.off()
png("/tmp/plots/03_y_pre_noise.png")
ggplot(df, aes(n, y)) + geom_point() + labs(title="y pre noise")
dev.off()

df = read.delim("/tmp/actions_post_noise", h=F, sep=" ", col.names=c("x", "y"))
png("/tmp/plots/03_x_y_scatter.png")
ggplot(df, aes(x, y)) + geom_point()
dev.off()
df$n = 1:nrow(df)
png("/tmp/plots/03_x_post_noise.png")
ggplot(df, aes(n, x)) + geom_point() + labs(title="x post noise")
dev.off()
png("/tmp/plots/03_y_post_noise.png")
ggplot(df, aes(n, y)) + geom_point() + labs(title="y post noise")
dev.off()

df = read.delim("/tmp/q_loss", h=F)
df$n = 1:nrow(df)
png("/tmp/plots/04_q_loss.png")
ggplot(df, aes(n, V1)) + geom_point() + 
  geom_smooth() + labs(title="q loss")
dev.off()

df = read.delim("/tmp/action_q_values", h=F)
summary(df)
df$n = 1:nrow(df)
png("/tmp/plots/05_action_q_values.png")
ggplot(df, aes(n, V1)) + geom_point() +
  geom_smooth() + labs(title="q values over time")
dev.off()

df = read.delim("/tmp/episode_len", h=F)
df$n = 1:nrow(df)
png("/tmp/plots/06_episode_len.png")
ggplot(df, aes(n, V1)) + geom_point() +
  geom_smooth() + labs(title="episode len")
dev.off()
