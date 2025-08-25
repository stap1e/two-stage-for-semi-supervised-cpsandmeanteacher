"""
 * URL: https://github.com/LiheYoung/UniMatch-V2
 * 
 * Copyright (c) LiheYoung
"""

# index from 0 to num_class-1
CLASSES = {
            'flare22': ['Foreground', 'Liver', 'Right kidney', 'Spleen', 'Pancreas', 'Aorta', 'IVC', 
                      'RAG', 'LAG', 'Gallbladder', 'Esophagus', 'Stomach', 'Duodenum', 'Left kidney'],
            'flare22_ssl': ['Liver', 'Right kidney', 'Spleen', 'Pancreas', 'Aorta', 'IVC', 
                      'RAG', 'LAG', 'Gallbladder', 'Esophagus', 'Stomach', 'Duodenum', 'Left kidney'],

            'amos': ['background', 'Liver', 'Right kidney', 'Spleen', 'Pancreas', 'Aorta', 'IVC', 
                      'RAG', 'LAG', 'Gallbladder', 'Esophagus', 'Stomach', 'Duodenum', 'Left kidney',
                      'Bladder', 'Prostate/Uterus'],
            'amos_ssl': ['Liver', 'Right kidney', 'Spleen', 'Pancreas', 'Aorta', 'IVC', 
                      'RAG', 'LAG', 'Gallbladder', 'Esophagus', 'Stomach', 'Duodenum', 'Left kidney',
                      'Bladder', 'Prostate/Uterus']
           }