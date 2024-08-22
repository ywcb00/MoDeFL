ROOTPATH = Sys.getenv(x="DFL_ROOT");
BASEPATH = paste(ROOTPATH, "scripts/plot", sep="/");

library("ggplot2");
library("ggrepel");
library("tidyr");

num_actors = 4;
learning_types = paste("dflv", 1:3, sep="");
learning_rates = paste("clr", c(0.1, 0.05, 0.02, 0.01, 0.005, 0.002), sep="");
ports = as.character(50504 + 1:num_actors);
collection_types = c("local/train", "local/eval", "neighbors/eval");
# metric_types = c("loss", "categorical_crossentropy", "categorical_accuracy");

collectStatsDataframe = function(logpath) {
  stats = data.frame();
  for(lt in learning_types) {
    lt_logpath = paste(logpath, lt, sep="/");
    for(lr in learning_rates) {
      lr_logpath = paste(lt_logpath, lr, sep="/");
      for(port in ports) {
        port_logpath = paste(lr_logpath, port, sep="/");
        for(ct in collection_types) {
          ct_logpath = paste(port_logpath, paste(ct, ".csv", sep=""), sep="/");
          tmp_df = read.table(paste(ct_logpath), sep=",", header=TRUE);
          tmp_df$epoch=1:nrow(tmp_df);
          tmp_df$collection_type=as.factor(ct);
          tmp_df$port=as.factor(port);
          tmp_df$learning_rate=as.factor(lr);
          tmp_df$learning_type=as.factor(lt);
          stats = rbind(stats, tmp_df);
        }
      }
    }
  }
  return (stats);
}

stats = collectStatsDataframe(paste(ROOTPATH, "log", sep="/"))

custom_palette = c("#7B1FA2", "#F57C00", "#303F9F", "#D01716");
pdf_path = file.path(ROOTPATH, "figures");
dir.create(pdf_path, showWarnings=FALSE);
for(lt in levels(stats$learning_type)) {
  pdf(file.path(pdf_path, paste(lt, "_gslr_plot.pdf", sep="")));
  for(lr in levels(stats$learning_rate)) {
    lr_plt = ggplot(data=stats[stats$learning_type==lt & stats$learning_rate==lr, ], mapping=aes(x=epoch, y=categorical_accuracy)) +
      geom_line(mapping=aes(color=collection_type, lty=port)) +
      scale_color_manual(values=custom_palette) +
      ggtitle(paste("Accuracy (lt=", lt, ", lr=", lr, ")", sep="")) +
      xlab("Epoch") + ylab("Categorical Accuracy");
    print(lr_plt);
  }
  dev.off();
}
