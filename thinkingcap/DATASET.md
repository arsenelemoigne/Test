# ThinkingCap training data (LUPIN pool dp13 and its top-ups)

What the trained models learn from: where the questions come from, how they were picked, how the answers and the
preference pairs are made, the rules, the counts, and real examples. Written 2 Nov 2026. Build history and every
iteration are in EXPERIMENT_LOG.md (29 Oct – 2 Nov); the pool-version table is in THINKINGCAP_FULL_DOSSIER.md.

## 1. The idea in one paragraph

The original Qwen3-0.6B answers each training question several times. Its answers are graded with the evaluation's own
graders (right / wrong / cut off at the token cap). From those graded answers we build **pairs**: a preferred answer
("winner") and a rejected one ("loser") for the same question. DPO then trains a small LoRA adapter to prefer winners
over losers. If the winner is usually the *shorter right* answer, the model learns to think less while staying right.

## 2. The questions

### 2.1 Pool dp13 (adopted 30 Oct): 11,999 questions, none from the test

| type (test part) | questions | sources (count) |
|---|---|---|
| Math (SVAMP) | 2,999 | MetaMath "easy" rewordings 1,865 · MetaMath solve-for-x / backwards 473 · GSM8K easy (1–2 calculations) 358 + 62 · GSM8K matched to SVAMP 241 |
| Code (MBPP+) | 3,001 | KodCode (MBPP-like slice: tasks ≤ 45 tokens, solutions ≤ 90; unit-test graded) |
| Instructions (IFEval) | 2,998 | built from human requests (no_robots) + IFEval constraints 1,828 · built from Dolly requests 653 · argilla ifeval-like 517 |
| Tools: one function | 1,426 | Glaive 849 · ToolACE 574 · Hermes 3 |
| Tools: several functions | 716 | ToolACE 716 |
| Tools: none fits | 859 | Glaive 434 · ToolACE 425 |

Source details (prepare_types.py):
- **gsm8k_easy**: GSM8K *train* problems whose solution makes 1–2 calculations (SVAMP's difficulty).
- **metamath_easy**: MetaMathQA rewordings of those easy GSM8K problems. **metamath_sv**: MetaMath "solve for x" and
  "asked backwards" rewrites (closest to SVAMP's twists). **gsm8k_matched**: GSM8K train matched to SVAMP's
  operation count, kind and length.
- **kodcode**: KodCode-Light-RL-10K, one-function easy/medium tasks with plain test functions turned into MBPP-style
  asserts; kept only if short like MBPP+.
- **ifeval_built**: a human-written request plus 1–3 IFEval-checkable constraints, drawn to match IFEval's own mix of
  constraint kinds and counts. **ifeval_like**: argilla/ifeval-like-data rows with IFEval's instruction ids.
- **tools**: ToolACE / Glaive / Hermes single-turn calls, cleaned to BFCL's format; "none fits" = a question offered a
  function that cannot answer it (Glaive no-fit verified by Haiku as truly unanswerable).

How the 11,999 were picked (build_pool.py): per test part, questions are drawn so their **prompt lengths match the
test's** (5 length bins of the choosing set); every candidate sharing an 8-word sequence with any test question is
dropped (contamination check); duplicates removed. Shares follow the test: 3,000 each for math, code and instructions,
3,000 split over the three tool parts like BFCL (400 / 200 / 240).

### 2.2 Top-ups (harder questions added later)

| top-up | added | questions | what |
|---|---|---|---|
| nofit1 | 30 Oct | 189 | no-fit questions the model sometimes gets wrong |
| tools1 | 30 Oct | 724 | hardest sometimes-right one-function (667) and several-function (57) questions |
| code1 | 30 Oct | 800 | hardest sometimes-right code tasks |
| nofit2 | 30 Oct | 552 | harder no-fit: offered a function ranked 4th–15th closest in meaning (Haiku-verified) |
| nofit4 | 1 Nov | 1,871 | our own hard no-fit set (xLAM irrelevance converted to BFCL); 257 sometimes wrong |
| math1 | 1 Nov | 2,000 | GSM8K + MetaMath (michal-bida's math sources), 10 answers each; 930 sometimes wrong |
| math2 | 1–2 Nov | 2,000 | same sources, the next 2,000 |
| instr2 | queued 2 Nov | 2,000 | michal-bida's instruction sources: allenai IF_multi_constraints_upto5 + RLVR-IFeval, as they come |
| code2 | queued 2 Nov | 1,500 | michal-bida's code selection: all of KodCode-V1 (Filter/Prefill/Leetcode/Evol), questions ≤ 700 characters, as they come |

## 3. The answers

The original Qwen3-0.6B (thinking on, the evaluation's sampling and token caps: 4,096; 8,192 for code) answers each
pool question **10 times** (lupin-answers-dp10, two halves on two GPUs). The bida-style runs use **4 answers per
question** (as michal-bida did). Every answer is graded: math by the final number, code by running the unit tests,
instructions by IFEval's checkers, tools by BFCL's checker. An answer cut off at the cap, or whose thinking never
closes, counts as **wrong**.

## 4. The pairs (lupin_pairs.py)

Per question, with p = right answers / all answers:

| kind | winner | loser | teaches |
|---|---|---|---|
| **guard** ("right beats wrong") | the shortest right answer | the longest wrong or capped answer | be right (and short) |
| **shorten** ("shorter right beats longer right") | the shortest right answer | the longest right answer, if ≥ 1.5× longer | think less |
| right (other recipes) | a right answer | a wrong one of similar length (within 25%) | be right, no length push |
| don't-over-cut (bidax5m/s only) | the original's median-length right answer | a trained model's wrong answer < 0.75× as long | think enough (failed: winner and loser by different models) |

Rules of the bida recipe (michal-bida's "Db"; our runs bidax2 – bidax7):
- always-right questions give only shorten pairs; never-right questions give none; sometimes-right ones give both;
- **balance**: per test part, no more shorten pairs than guard pairs (shorten pairs randomly dropped);
- no difficulty rule, no matching of pairs to the test (`--lp-no-match`), no copying term (`--lp-nll 0`).

Earlier recipes (l2s50e, Run A/B, per-type merges) added a difficulty rule (no shorten pair under 50% solved), easier
shorten on always-solved questions (from 1.2×), up to 2 shorten per guard, pair matching to the test's
type × difficulty × length cells, and a copying term.

### Pair counts (bidax4 = bidax6's pairs: dp13 + nofit1, tools1, code1, nofit2, nofit4; 4 answers)

| part | guard | shorten | total |
|---|---|---|---|
| math | 475 | 475 | 950 |
| code | 886 | 886 | 1,772 |
| instructions | 681 | 681 | 1,362 |
| tools: one function | 229 | 229 | 458 |
| tools: several | 174 | 174 | 348 |
| tools: none fits | 260 | 260 | 520 |
| **total** | | | **5,410** (5,362 trained + 48 held out) |

michal-bida's own Db: 5,862 pairs (guard per domain: code 941, instructions 724, math 696, tools 570; shorten capped to match).

## 5. Training settings (dpo.py)

DPO (β 0.1) on a LoRA adapter (rank 32, alpha 64, dropout 0, all 7 projection layers), AdamW 2e-5, 16 pairs per step,
one pass, the final adapter scored. Ours: bfloat16, cosine to 2e-6 after 10 warm-up steps, 48 pairs held out.
michal-bida's (`--bida-exact`): float32 base + float16 compute with a loss scaler, warm-up 5% of steps, cosine to 0,
no pair held out, pairs over 8,704 tokens skipped.

## 6. How it differs from michal-bida's data

| | ours | his |
|---|---|---|
| question selection | matched to the test's prompt lengths | as they come (8-gram contamination check only) |
| math | mostly *easy* MetaMath / GSM8K (1–2 calculations), SVAMP-like | GSM8K + MetaMath, all difficulties |
| code | short KodCode-Light tasks (≤ 45 tokens) | all KodCode-V1, questions up to 700 characters |
| instructions | built from human requests + IFEval constraints; argilla | allenai multi-constraint + RLVR-IFeval |
| tools | ToolACE / Glaive / Hermes, ~39% no-fit pairs | ToolACE + xLAM, ~45% no-fit pairs |

## 7. Real examples (bidax4's pairs, decoded)

Picked automatically: per type and kind, a short pair (prompt under 400 tokens; the lower quarter by length).
Thinking shown from its start only.

### Math (SVAMP)

**"right beats wrong" (guard)** — winner 369 tokens, loser 402 tokens

Question:

    If John uses a 75-watt electric fan for 8 hours each day, what is the total amount of kWh of electric energy he consumes in a month (30 days) from using the fan?
    
    The last line of your reply must start with 'Final answer:' followed by the number alone.

Winner, thinking (start):

    Okay, let's see. The problem is about calculating the total electric energy John consumes in a month using a 75-watt fan for 8 hours each day. The month has 30 days. Hmm, I need to find out how much energy he uses in kWh.
    
    First, I remember that power in electrical energy is calculated by multiplying watts (W) by hours (h) to get kilowatt-hours (kW […]

Winner, answer:

    Final answer: 18

Loser, thinking (start):

    Okay, let's see. John uses a 75-watt electric fan for 8 hours each day for 30 days. I need to find out how much electric energy he consumes in a month. Hmm, energy consumption in electrical terms... I remember that power is measured in watts, and time in hours, so kWh is kilowatt-hour. 
    
    First, I should calculate the daily energy consumption. Power […]

Loser, answer:

    Final answer: 18000

**"shorter right beats longer right" (shorten)** — winner 288 tokens, loser 468 tokens

Question:

    Michael has 4 packs of crayons and wants to buy 2 more. One pack of crayons costs $2.5. How much are all the packs of crayons Michael will have after the purchase worth?
    
    The last line of your reply must start with 'Final answer:' followed by the number alone.

Winner, thinking (start):

    Okay, let's see. Michael has 4 packs of crayons and wants to buy 2 more. Each pack costs $2.5. I need to find out how much all the packs will be worth after buying those 2. Hmm.
    
    First, let me add up the number of packs he currently has. He has 4, and he wants to buy 2 more. So total packs would be 4 + 2, which is 6 packs. Then each pack is $2.5, s […]

Winner, answer:

    Final answer: 15

Loser, thinking (start):

    Okay, let's see. Michael has 4 packs of crayons and wants to buy 2 more. Each pack costs $2.5. I need to find out how much all the packs will be worth after buying the new ones. Hmm, first, let me break it down step by step.
    
    So, initially, Michael has 4 packs. Then he buys 2 more. That means he ends up with 4 plus 2, right? So 4 + 2 equals 6 packs […]

Loser, answer:

    Final answer: 15

### Code (MBPP+)

**"right beats wrong" (guard)** — winner 489 tokens, loser 660 tokens

Question:

    Write a complete Python solution to the following task. Return the solution in one Python code block.
    
    """
    You are given an array of integers where every element appears twice except for one. Find that single one.
    
    Example
    
    Input
    
    4 1 2 1 2
    
    Output
    
    4
    assert single_number([4, 1, 2, 1, 2]) == 4
    """
    
    The tests call `single_number` and nothing else, so any other function name fails. Define `single_number`, spelled exactly like that.

Winner, thinking (start):

    Okay, I need to solve this problem where I have an array of integers, every element appears twice except one. My task is to find that single one. Let's think about how to approach this.
    
    Hmm, the straightforward way is to use a hash map. Because if I count each number's occurrences, the one that appears once will be the answer. But wait, what about […]

Winner, answer:

    ```python
    def single_number(arr):
        result = 0
        for num in arr:
            result ^= num
        return result
    ```

Loser, thinking (start):

    Okay, I need to solve this problem where I have an array of integers, and every element appears exactly twice except for one. My task is to find that single one. Let's think about how to approach this.
    
    Hmm, the classic problem is the "single number" problem, right? Oh right, there's a trick with XOR. Because XOR has some properties here. Let me re […]

Loser, answer:

    ```python
    def single_number(nums):
        return sum(nums) ^ sum(nums)
    ```

**"shorter right beats longer right" (shorten)** — winner 424 tokens, loser 694 tokens

Question:

    Write a complete Python solution to the following task. Return the solution in one Python code block.
    
    """
    Write a function to calculate the total number of unique words in a given string.
    assert unique_word_count("hello world") == 2
    """
    
    The tests call `unique_word_count` and nothing else, so any other function name fails. Define `unique_word_count`, spelled exactly like that.

Winner, thinking (start):

    Okay, I need to write a Python function called unique_word_count that takes a string and returns the number of unique words. Let me think about how to approach this.
    
    First, I remember that in Python, strings are immutable, so I can't just modify them directly. So the plan is to split the string into words, then remove duplicates, and count them. […]

Winner, answer:

    ```python
    def unique_word_count(s):
        words = s.split()
        return len(set(words))
    ```

Loser, thinking (start):

    Okay, I need to write a Python function called unique_word_count that takes a string and returns the total number of unique words. Let me think about how to approach this.
    
    First, I remember that in Python, strings are immutable, so I can't just modify them. So the first step is to split the string into individual words. The split method can be use […]

Loser, answer:

    ```python
    def unique_word_count(s):
        return len(set(s.split()))
    ```

### Instructions (IFEval)

**"right beats wrong" (guard)** — winner 298 tokens, loser 340 tokens

Question:

    Which city is the best to relocate in Texas Your entire response should be in English, and in all capital letters.

Winner, thinking (start):

    Okay, the user is asking which city is the best to relocate in Texas. They want the answer in all capital letters. Let me think about this.
    
    First, I need to consider factors like job opportunities, cost of living, proximity to home, and personal preferences. Texas has a lot of cities, so maybe the answer is one of the most popular or well-known on […]

Winner, answer:

    BEST CITY IN TEXAS IS AUSTIN.

Loser, thinking (start):

    Okay, the user is asking which city is the best to relocate in Texas. They want the answer in all capital letters. Let me think about this.
    
    First, I need to recall the top cities in Texas. Cities like Houston, Dallas, Austin, and San Antonio come to mind. Each has different factors like population, job markets, cost of living, and infrastructure. […]

Loser, answer:

    HONOLULU, DALLAS, AUSTIN, SAN ANTONIO

**"shorter right beats longer right" (shorten)** — winner 210 tokens, loser 418 tokens

Question:

    What should I do during a visit to Indianapolis? Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.

Winner, thinking (start):

    Okay, the user is asking what to do during a visit to Indianapolis. Let me start by recalling what Indianapolis is known for. It's a major city in the Midwest, so maybe cultural and historical sites. I should mention landmarks like the Indianapolis Zoo, the Indiana University, and the Indianapolis River. 
    
    Wait, the user wants the answer wrapped in […]

Winner, answer:

    <<A City of Culture and Tradition>>  
    Visit the Indianapolis Zoo, explore Indiana University, and enjoy the vibrant city life. Wander through historic neighborhoods and enjoy local culture.

Loser, thinking (start):

    Okay, the user is asking what to do during a visit to Indianapolis. First, I need to consider the key points of Indianapolis. It's a city in the US, so the main attractions would be the landmarks, cultural sites, and natural areas. Let me start by listing the major attractions.
    
    First, the city's historical significance. Maybe mention the Jefferson […]

Loser, answer:

    <<Discovering the Heart of Indianapolis>>  
    Visit the Jefferson Memorial, Indiana State Capitol, and Indianapolis Zoo to experience the city’s rich history and natural beauty. Explore the Indiana Dunes, enjoy the Indianapolis River, and attend local cultural events. Indianapolis is a vibrant mix of culture, history, and nature, making it a must-visit destination.

### Tools: one function (BFCL simple)

**"right beats wrong" (guard)** — winner 117 tokens, loser 149 tokens

Question:

    Hi, I need to convert 5 miles into kilometers. Can you help me with that?

Winner, thinking (start):

    Okay, the user wants to convert 5 miles into kilometers. Let me check the available functions. There's a convert_length function that takes length, from_unit, and to_unit. The parameters are required, so I need to fill in the length as 5, from_unit as "miles", and to_unit as "kilometers". I should make sure the units are correctly specified. No oth […]

Winner, answer:

    [convert_length(length=5, from_unit="miles", to_unit="kilometers")]

Loser, thinking (start):

    Okay, the user wants to convert 5 miles into kilometers. Let me check the functions available. There's a convert_length function. It requires length, from_unit, and to_unit. The user provided 5 miles, so the length is 5. The from_unit is miles, and the to_unit is kilometers. I need to make sure the parameters are correctly set. The function's param […]

Loser, answer:

    [convert_length(length=5, from_unit="miles", to_unit="km")]

**"shorter right beats longer right" (shorten)** — winner 95 tokens, loser 203 tokens

Question:

    Can you help me find a note in my collection? I think the ID is 123ABC.

Winner, thinking (start):

    Okay, the user wants to find a note with ID 123ABC. Let me check the available functions. There's a get_note function that requires a note_id. The parameters needed are present here, so I should call that function with note_id=123ABC. No other functions are needed. I just need to make sure the ID is correctly formatted as a string.

Winner, answer:

    [get_note(note_id="123ABC")]

Loser, thinking (start):

    Okay, the user is asking if I can help them find a note in their collection with the ID 123ABC. Let me check the available functions.
    
    Looking at the functions provided, there's one called get_note. Its description says it retrieves a specific note by its ID. The parameters required are a note_id, which is a string. The user provided the ID as 123A […]

Loser, answer:

    [get_note(note_id="123ABC")]

### Tools: several functions (BFCL multiple)

**"right beats wrong" (guard)** — winner 151 tokens, loser 263 tokens

Question:

    Can you check if this LinkedIn prostring is open: https://www.linkedin.com/in/johndoe?

Winner, thinking (start):

    Okay, the user wants to check if a LinkedIn prostring is open at the URL https://www.linkedin.com/in/johndoe. Let me look at the available functions. There's a function called Get_Open_Prostring_Status that takes a LinkedIn prostring URL. The parameters required are the linkedin_url. The user provided exactly that URL, so I need to call this functi […]

Winner, answer:

    [Get_Open_Prostring_Status(linkedin_url="https://www.linkedin.com/in/johndoe")]

Loser, thinking (start):

    Okay, the user wants to check if a LinkedIn prostring is open. The URL provided is https://www.linkedin.com/in/johndoe. Let me look at the available functions.
    
    The functions available are Generate_Lorem_Ipsum, Get_Fear_and_Greed_Index, Retrieve_Rewards, and Get_Open_Prostring_Status. The user is asking about a prostring's open status, so the relev […]

Loser, answer:

    [Get_Open_Prostring_Status(linkedin_url="https://www.linkedin.com/in/johndoe?")]

**"shorter right beats longer right" (shorten)** — winner 132 tokens, loser 237 tokens

Question:

    I'm interested in betting on a horse race next weekend and want to study trainers' recent performance. Can you get me the win rates for the trainers in the last 30 days?

Winner, thinking (start):

    Okay, the user wants to bet on a horse race next weekend and study trainers' recent performance. They mentioned wanting the win rates for trainers in the last 30 days.
    
    Looking at the available functions, there's Get_Trainers__Win_Rate which requires the last_days parameter. The user specified "last 30 days," so I need to set last_days=30. The othe […]

Winner, answer:

    [Get_Trainers__Win_Rate(last_days=30)]

Loser, thinking (start):

    Okay, the user wants to bet on a horse race next weekend and study trainers' recent performance. They specifically mentioned wanting the win rates for trainers in the last 30 days.
    
    Looking at the available functions, there's one called Get_Trainers__Win_Rate. Its description says it retrieves win rates for trainers in horse racing along with runs […]

Loser, answer:

    [Get_Trainers__Win_Rate(last_days=30)]

### Tools: none fits (BFCL irrelevance)

**"right beats wrong" (guard)** — winner 154 tokens, loser 226 tokens

Question:

    Can you get me the list of categories related to American Football?

Winner, thinking (start):

    Okay, the user is asking for the list of categories related to American Football. Let me check the available functions. The only function provided is Explore_Areas, which retrieves areas related to soccer, teams, or leagues. The parameters needed are area_type and location. But the user mentioned American Football, which is a sport, not a stadium o […]

Winner, answer:

    None of the provided functions can be used to retrieve categories related to American Football.

Loser, thinking (start):

    Okay, the user is asking for the list of categories related to American Football. Let me check the available functions. There's the Explore_Areas function, which can retrieve areas like stadiums, teams, or leagues. The parameters required are area_type and optionally location. The user mentioned American Football, but the function's area_type param […]

Loser, answer:

    [Explore_Areas(area_type="league", location="American Football")]

**"shorter right beats longer right" (shorten)** — winner 142 tokens, loser 219 tokens

Question:

    What details can you provide about the IP addresses '74.125.200.100' and '209.85.247.100'?

Winner, thinking (start):

    Okay, the user is asking for details about two IP addresses: '74.125.200.100' and '209.85.247.100'. Let me check the available functions. The only function provided is get_callback_url_status, which checks a callback URL's POST ability. But the user is asking about IP addresses, not a URL. There's no function here that can retrieve IP information. […]

Winner, answer:

    The provided functions cannot retrieve information about IP addresses.

Loser, thinking (start):

    Okay, let's see. The user is asking for details about two IP addresses: '74.125.200.100' and '209.85.247.100'. But looking at the functions provided, the only one available is get_callback_url_status, which checks the status of a callback URL. The parameters needed for that function are a callback URL, not IP addresses.
    
    Hmm, so the user's question […]

Loser, answer:

    None of the provided functions can be used to retrieve details about IP addresses.

