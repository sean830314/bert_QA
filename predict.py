from transformers import BertForQuestionAnswering, BertTokenizer
import torch
def ask(context, question):
    model = BertForQuestionAnswering.from_pretrained('debug_squad/model')
    # model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('debug_squad/model')
    question = question
    answer_text = context
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    print('The input has a total of {:} tokens.'.format(len(input_ids)))
    # BERT only needs the token IDs, but for the purpose of inspecting the
    # tokenizer's behavior, let's also get the token strings and display them.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # For each token and its id...
    for token, id in zip(tokens, input_ids):
        # If this is the [SEP] token, add some space around it to make it stand out.
        if id == tokenizer.sep_token_id:
            pass
            # print('')

        # Print the token string and its ID in two columns.
        # print('{:<12} {:>6,}'.format(token, id))

        if id == tokenizer.sep_token_id:
            pass
            # print('')

    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)
    attention_mask = [1  if input_ids != -1 else 0 for i in input_ids ]
    # Run our example through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]),  token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Combine the tokens in the answer and print it out.
    answer = ' '.join(tokens[answer_start:answer_end+1])
    print("Question is : ", question)
    print()
    print('Answer: "' + answer + '"')
context = "CBS broadcast Super Bowl 50 in the U.S., and charged an average of $5 million for a 30-second commercial during the game. The Super Bowl 50 halftime show was headlined by the British rock group Coldplay with special guest performers Beyonc\u00e9 and Bruno Mars, who headlined the Super Bowl XLVII and Super Bowl XLVIII halftime shows, respectively. It was the third-most watched U.S. broadcast ever."
questions =  ["Which network broadcasted Super Bowl 50 in the U.S.?", "What was the average cost for a 30 second commercial during Super Bowl 50?"]
print("Context is : ", context)
print()
for question in questions:
    ask(context,question)
