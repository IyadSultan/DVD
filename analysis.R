ds<- read.csv("modified_notes_4o_11_to_25.csv")
print(dim(ds))
print(colnames(ds))
#  [1] "original_note_number" "new_note_name"        "question"
#  [4] "source_document"      "ideal_answer"         "correct_answer"
#  [7] "ai_answer"            "note_answer"          "timestamp"
# [10] "total_tokens"
library(dplyr)
ds %>%
    mutate(ai_self_failre=ifelse(new_note_name=="AI" & 
            source_document=="AI" & ai_answer!=correct_answer,1,0)) %>%
    mutate(notes_self_failre=ifelse(new_note_name!="AI" & 
            source_document!="AI" & note_answer!=correct_answer,1,0)) %>%
    mutate(corrected_ai_mistake=ifelse(ai_answer!=correct_answer & !notes_self_failre,1,0)) %>%
    mutate(corrected_note_mistake=ifelse(note_answer!=correct_answer & !ai_self_failre,1,0)) %>%
    group_by(new_note_name) %>%
    summarise(
        ai_correct = sum(correct_answer == ai_answer),
        note_correct = sum(correct_answer == note_answer),
        ai_not_know=sum(ai_answer=="E"),
        note_not_know=sum(note_answer=="E"),
        ai_mistakes=sum(correct_answer!=ai_answer & ai_answer!="E"),
        note_mistakes=sum(correct_answer!=note_answer & note_answer!="E"),
        ai_corrected_mistakes=sum(corrected_ai_mistake),
        note_corrected_mistakes=sum(corrected_note_mistake),
        ai_self_mistakes=sum(ai_self_failre),
        note_self_mistakes=sum(notes_self_failre),
        total_ai=n(),
        total_note=n()
    ) %>% 
    mutate(ai=ai_correct/total_ai,note=note_correct/total_note) %>%
    mutate(ai_corrected_succss=(total_ai-ai_corrected_mistakes)/total_ai,
        note_corrected_succss=(total_note-note_corrected_mistakes)/total_note) %>%
    View()



