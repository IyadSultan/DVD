df<-read.csv("modified_notes/modified_notes.csv")
colnames(df)
#  [1] "original_note_number" "new_note_name"        "modified_text"
#  [4] "modifications"        "processing_time"      "input_tokens"
#  [7] "output_tokens"        "total_tokens"         "added_text"
# [10] "removed_text"         "model"                "timestamp"

table(df$new_note_name)
