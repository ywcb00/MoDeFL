ROOTPATH = Sys.getenv(x="DFL_ROOT");
BASEPATH = paste(ROOTPATH, "scripts/plot", sep="/");

library("ggplot2");
library("ggrepel");
library("tidyr");

learning_rates = c(0.1, 0.05, 0.02, 0.01);
ports = c(50505, 50506, 50507, 50508);

custom_palette = c("#7B1FA2", "#F57C00", "#303F9F", "#D01716");

pdf(paste(BASEPATH, "figures", "gslr_plot.pdf", sep="/"));

metrics = list();
for(lr in paste("lr", learning_rates, sep="")) {
  metrics[[lr]] = list()

  for(port in ports) {
    actor_log_dir = paste(ROOTPATH, "log/gslr", lr, port, sep="/");

    data_train = read.table(paste(actor_log_dir, "local/train.csv", sep="/"), sep=",", header=TRUE);
    data_eval = read.table(paste(actor_log_dir, "local/eval.csv", sep="/"), sep=",", header=TRUE);

    data_neighbors_eval = read.table(paste(actor_log_dir, "neighbors/eval.csv", sep="/"), sep=",", header=TRUE);

    metrics[[lr]][[paste("port", port, sep="")]] = list()
    for(cn in colnames(data_train)) {
      metrics[[lr]][[paste("port", port, sep="")]][[cn]] =
        data.frame(idx=1:length(data_train[[cn]]), train=data_train[[cn]],
         eval=data_eval[[cn]], neighbor_eval=data_neighbors_eval[[cn]]);
    }
  }

  lr_plot_data = data.frame();
  actor_plots = vector("list", length(ports));
  for(counter in 1:length(ports)) {
    port = ports[counter];
    metric_df = metrics[[lr]][[paste("port", port, sep="")]][["categorical_accuracy"]]
    plot_data = metric_df %>% pivot_longer(cols=colnames(metric_df)[-1], names_to='type', values_to='value');
    lr_plot_data = rbind(lr_plot_data, data.frame(plot_data, port=as.factor(port)));
    actor_plots[[counter]] = ggplot(data=plot_data, mapping=aes(x=idx, y=value)) +
      geom_line(mapping=aes(color=type)) +
      scale_color_manual(values=custom_palette) +
      ggtitle(paste("Accuracy (lr=", lr, ", port=", port, ")", sep="")) +
      xlab("Epoch") + ylab("Categorical Accuracy");
  }
  for(plt in actor_plots) {
    print(plt);
  }
  lr_plt = ggplot(data=lr_plot_data, mapping=aes(x=idx, y=value)) +
    geom_line(mapping=aes(color=type, lty=port)) +
    scale_color_manual(values=custom_palette) +
    ggtitle(paste("Accuracy (lr=", lr, ")", sep="")) +
    xlab("Epoch") + ylab("Categorical Accuracy");
  print(lr_plt);
}

dev.off();
