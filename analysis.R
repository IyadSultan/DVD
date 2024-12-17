ds<- read.csv("results_gpt_4o.csv")
print(dim(ds))
print(colnames(ds))
# [1] "original_note_number" "new_note_name"        "question"
# [4] "correct_answer"       "ai_answer"            "note_answer"
library(dplyr)
ds %>%
    group_by(new_note_name) %>%
    summarise(
        ai_correct = sum(correct_answer == ai_answer),
        note_correct = sum(correct_answer == note_answer),
        ai_not_know=sum(ai_answer=="E"),
        note_not_know=sum(note_answer=="E"),
        ai_mistakes=sum(correct_answer!=ai_answer)-ai_not_know,
        note_mistakes=sum(correct_answer!=note_answer)-note_not_know,
        total_ai=n(),
        total_note=n()
    ) %>% 
    mutate(ai=ai_correct/total_ai,note=note_correct/total_note) %>%
    View()



