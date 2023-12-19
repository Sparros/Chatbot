# Load the knowledge base from a CSV file
read_expr = Expression.fromstring
kb = []
with open('assignment\\kb.csv', 'r') as data:
    kb = [read_expr(row) for row in data.readlines()]
# Check KB for consistency/ no contradiction
prover = ResolutionProver()
contradiction = []
consistent = []
for expr in kb:
    if prover.prove(Expression.fromstring(f'~({expr})')):
        contradiction.append(expr)
        #print(f"Contradiction found with {expr}")
    else:
        consistent.append(expr)
        #print(f"{expr} is consistent with the KB")
if len(contradiction) >= 1: 
    for expr in contradiction:
        print(f"Contradiction found with {expr}")

# Here are the processing of the new logical component:
elif cmd == 31: # if input pattern is "I know that * is *"
    object,subject=params[1].split(' is ')
    expr=read_expr(subject + '(' + object + ')')
    # >>> ADD SOME CODES HERE to make sure expr does not contradict 
    if prover.prove(Expression.fromstring(f'~({expr})')):
    # with the KB before appending, otherwise show an error message.
        kb.append(expr) 
    else:
        print('Contradiction with the knowledge base')
    print('OK, I will remember that',object,'is', subject)
elif cmd == 32: # if the input pattern is "check that * is *"
    object,subject=params[1].split(' is ')
    expr=read_expr(subject + '(' + object + ')')
    answer=ResolutionProver().prove(expr, kb, verbose=True)
    if answer:
    print('Correct.')
    else:
    print('It may not be true.') 
    # >> This is not an ideal answer.
    # >> ADD SOME CODES HERE to find if expr is false, then give a
    # definite response: either "Incorrect" or "Sorry I don't know." 

elif userInput.upper() == "WHAT DO YOU KNOW":
    print('Here is what I know')
    for expr in kb:
        print(expr);



<!-- Logic input -->
<category><pattern> I KNOW THAT * IS *</pattern>
  <template>#31$<star index="1"/> is <star index="2"/></template></category>  

<category><pattern> CHECK THAT * IS *</pattern>
  <template>#32$<star index="1"/> is <star index="2"/></template></category>  