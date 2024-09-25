# ---------------- 模型部分 --------------------
class MultiTaskBERTModel(nn.Module):
    def __init__(self, num_labels_ner, num_relations, num_classes):
        super(MultiTaskBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("path/to/google-bert/bert-base-uncased")
        self.num_ner = num_labels_ner
        self.num_relations = num_relations
        self.num_classes = num_classes

        self.subject_classifier = nn.Linear(self.bert.config.hidden_size, 2)  # Subject: 0 or 1
        self.object_classifier = nn.Linear(self.bert.config.hidden_size, 2)   # Object: 0 or 1
        self.relation_classifier = nn.Linear(self.bert.config.hidden_size * 2, self.num_relations)  # RE classifier
        self.classifier_ner = nn.Linear(self.bert.config.hidden_size, self.num_ner)  # NER classifier
        self.classifier_re = nn.Linear(self.bert.config.hidden_size, self.num_classes)  # RE head
        self.classifier_classification = nn.Linear(self.bert.config.hidden_size, self.num_classes)  # Classification head

    def forward(self, input_ids, attention_mask, task_type):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        if task_type == 'ner':
            logits = self.classifier_ner(sequence_output)
            return logits
        elif task_type == 'triple':
            subject_logits = self.subject_classifier(sequence_output)
            object_logits = self.object_classifier(sequence_output)
            return subject_logits, object_logits, sequence_output
        elif task_type == 're':
            logits = self.classifier_re(sequence_output[:, 0, :])  # Using [CLS] token embedding for RE classification
            return logits
        elif task_type == 'tc':
            logits = self.classifier_classification(sequence_output[:, 0, :])  # Using [CLS] token embedding for classification
            return logits

    def predict_relations(self, self, sequence_output, subject_position, object_position):
        subject_embedding = sequence_output[subject_position, :]
        object_embedding = sequence_output[object_position, :]
        combined_embeddings = torch.cat([subject_embedding, object_embedding], dim=-1)
        relation_logits = self.relation_classifier(combined_embeddings)
        return relation_logits
