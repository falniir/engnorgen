import ludwig
from ludwig import visualize

test_stats_per_model = ludwig.test(
  model_path='results/experiment_run/model',
  data_csv='data/test.csv'
)

metadata = ludwig.collect_metadata(
    test_stats_per_model,
    'output',
    'combined_input'
    )

output_feature_name = 'output'
top_n_classes = 3
normalize = True


ludwig.visualize.confusion_matrix(
  test_stats_per_model,
  metadata,
  output_feature_name,
  top_n_classes,
  normalize,
  model_names=None,
  output_directory=None,
  file_format='pdf'
)