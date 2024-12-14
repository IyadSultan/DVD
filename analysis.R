ds<- read.csv("results_dvd_evaluator.csv")
print(dim(ds))
print(colnames(ds))
# [1] "original_note_number" "new_note_name"        "question"
# [4] "correct_answer"       "ai_answer"            "note_answer"
library(dplyr)
ds %>%
    group_by(original_note_number) %>%
    group_by(new_note_name) %>%
    summarise(
        ai_correct = sum(correct_answer == ai_answer),
        note_correct = sum(correct_answer == note_answer),
    ) %>% View()

