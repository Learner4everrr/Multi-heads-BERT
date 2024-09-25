# -------------------- Alternating training function ---------------------
def train_alternating(model, train_loader_ner, train_loader_triple, train_loader_re, train_loader_tc, optimizer, num_epochs):
    model.train()
    iter_ner = iter(train_loader_ner)
    iter_triple = iter(train_loader_triple)
    iter_re = iter(train_loader_re)
    iter_tc = iter(train_loader_tc)

    for epoch in range(num_epochs):
        step = 0
        progress_bar = tqdm(range(len(train_loader_ner)), desc=f"Epoch {epoch + 1}")

        for _ in progress_bar:
            # ---------------- Triple Extraction ----------------
            batch = get_next_batch(iter_triple, train_loader_triple)
            optimizer.zero_grad()

            # Extract necessary values from batch
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            subject_labels = batch['subject_labels'].to("cuda")
            object_labels = batch['object_labels'].to("cuda")

            subject_logits, object_logits, sequence_output = model(input_ids=input_ids, attention_mask=attention_mask, task_type='triple')
            loss_triple = compute_loss_triple(subject_logits, object_logits, subject_labels, object_labels)

            relation_logits = []
            relation_labels = []

            for idx in range(len(batch['relations'])):
                relations = batch['relations'][idx]
                for subj_pos, obj_pos, relation in relations:
                    relation_logit = model.predict_relations(sequence_output[idx], subj_pos, obj_pos)
                    relation_logits.append(relation_logit)
                    relation_labels.append(relation_to_index[relation])  # Convert relation to index

            if len(relation_logits) > 0:
                relation_logits = torch.stack(relation_logits)  # Stack relation logits
                relation_labels = torch.tensor(relation_labels).to("cuda")

            # Compute loss
            loss_triple = compute_loss_triple(subject_logits, object_logits, subject_labels, object_labels, relation_logits, relation_labels)
            loss_triple.backward()
            optimizer.step()

            # ---------------- Relation Extraction ----------------
            batch_re = get_next_batch(iter_re, train_loader_re)
            optimizer.zero_grad()
            input_ids = batch_re[0].to("cuda")
            attention_mask = batch_re[1].to("cuda")
            labels = batch_re[2].to("cuda")
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, task_type='re')
            loss_re = criterion_re(outputs, labels)
            loss_re.backward()
            optimizer.step()

            # ---------------- Usage Classification ----------------
            batch_tc = get_next_batch(iter_tc, train_loader_tc)
            optimizer.zero_grad()
            input_ids = batch_tc[0].to("cuda")
            attention_mask = batch_tc[1].to("cuda")
            labels = batch_tc[2].to("cuda")
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, task_type='tc')
            loss_tc = criterion_tc(outputs, labels)
            loss_tc.backward()
            optimizer.step()

            # ---------------- NER ----------------
            batch = get_next_batch(iter_ner, train_loader_ner)
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to("cuda"), attention_mask.to("cuda"), labels.to("cuda")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, task_type='ner')
            loss_ner = compute_loss_ner(outputs, labels, model.num_ner)
            loss_ner.backward()
            optimizer.step()


            step += 1
            progress_bar.set_postfix({"NER Loss": loss_ner.item(), "Triple Loss": loss_triple.item(), "RE loss": loss_re.item(), "Usage Classification": loss_tc.item()})
            # progress_bar.set_postfix({"NER Loss": loss_ner.item()})
            # progress_bar.set_postfix({"Triple Loss": loss_triple.item()})
            # progress_bar.set_postfix({"RE Loss": loss_re.item()})
            # progress_bar.set_postfix({"Usage Classification": loss_tc.item()})

            # if step % 10 == 0:
            #    break

            # ---------------- Evaluation ----------------
            # print("Evaluating")
            # eval_loss_ner = eval_ner(model, train_loader_ner)
            # print(eval_loss_ner)
            # break

            eval_ner(model, test_loader_ner)
            eval_triple(model, test_loader_triple)
            eval_classification(model, test_loader_re, task_type='re')
            eval_classification(model, test_loader_uc, task_type='tc')

