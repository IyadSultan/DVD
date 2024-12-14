ds<- read.csv("E:/Dropbox/AI/Projects/DVD/results_note2.csv")
print(dim(ds))
# [1] "folder_name"     "note_name"       "question"        "best_answer"    
# [5] "correct_answer"  "ai_answer"       "note_answer"     "ai_word_count"  
# [9] "note_word_count"
library(dplyr)
ds %>%
    group_by(folder_name) %>%
    group_by(note_name) %>%
    summarise(
        ai_correct = sum(best_answer == ai_answer),
        note_correct = sum(best_answer == note_answer)
    ) %>% View()

