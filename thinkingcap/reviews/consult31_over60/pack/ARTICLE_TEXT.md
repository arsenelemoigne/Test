16 November 2026 · BottleCap AI challenge · Read full screen ⤢

The small model already knew the short answer

I taught Qwen3-0.6B to think 55% less for less than 1 point, using online DPO.

54.8%fewer output tokens

−0.78points of accuracy, averaged over math, code, instructions and tool calls

Online DPOone training run, from start to end

AbstractCopy

Small reasoning models think far more than they need to. I trained one to think 55% less, at −0.8 points of accuracy, and then analyzed where the savings come from.

The question. BottleCap's challenge: "make Qwen3-0.6B think less without losing accuracy". My personal target: more than 50% fewer tokens at better than −1 point, on a full test of 2,759 questions answered 8 times each (math, code, instructions, tool calls). Switching thinking off saves 94% but costs 22.3 points.

Diagnosis.

The model writes about 670 thinking tokens for every 69 tokens of answer.

13.5% of its answers get stuck in a loop, and they hold half of everything it writes.

Wrong answers run 3 times longer than right ones.

Thinking pays very unevenly across task types.

Method. Online DPO. Each round the model answers fresh questions, learns from pairs of its own answers, and is checked on a hold-out sample. The pair rule, in one line:
          right ≻ wrong · shorter right ≻ longer right (by 1.3×) · except: floor and cap per task type
          Every round is measured against the original model, and a floor, a cap and a brake per task type stop the shortening before accuracy falls.

Results. 54.8% fewer tokens at −0.78 points; a second run with new seeds lands at 52.1% / −0.58. Per task type, the average hides a cost on math (−2.8) and code (−2.0), offset by "no function fits" (+5.9). For comparison, asking for brief thinking in the prompt saves 22% at −2.7. And question by question, some easy questions it used to solve now sometimes come out wrong. Which raises the question: is it shorter because it got smarter?

Did it get smarter? Not really. What happened underneath the average: question by question, about 5 answers in 100 went from right to wrong and 4 from wrong to right. The answers moved, but the average stays flat.

It lost consistency. Each question is asked 8 times. On math, the questions it solves all 8 times drop from 70% to 62%: it still knows how, but now sometimes skips a check it needed.

It cut steps, not words. A model can think less by dropping steps (trimming) or by writing the same steps in fewer words (condensing). Ours almost only trimmed: at least 90% of the saving is steps dropped. The steps it kept are as long and as detailed as before.

It learned when to stop. The original finds its answer about halfway through its thinking, then keeps re-checking it. Two-thirds of what training cut was that re-checking. The marginal value of a thinking token falls as the thinking goes on: the tokens it cut were worth about 0.2 points of accuracy per 100, twenty times less than an average thinking token.

It generalizes. On four benchmarks it never trained on, it still saves 52% to 63% of its tokens, with accuracy within a few points. And with thinking switched off, the online-trained models answer about 2 points better than the original: training improved the model itself a little.

Making a small model think less

1. Why thinking length matters

A language model handles three kinds of tokens:

Input tokens: the prompt. They are fixed by the task.

Thinking tokens: the reasoning a "thinking" model writes between &lt;think&gt; tags before answering.

Answer tokens: what the user actually receives.

The challenge counts the last two. For Qwen3-0.6B1 the split is lopsided: on average 670 thinking tokens for 69 answer tokens, so 91% of what it writes is thinking.

Original 739

After training 321

        thinkingthe answerone square = 10 tokens

Why does thinking help at all? A model like this is trained to reason in writing: first by imitation on worked solutions, then by reinforcement on whether its final answer is right. Each token is sampled from a probability distribution, and a longer trace explores more paths: a different way to set up the problem, a re-check that catches a slip, a second attempt after a dead end. Somewhere along the way a useful token becomes likely, and the answer improves. Thinking is search, written out.

Experiments show that this search has diminishing returns: past a point, longer thinking stops improving the answer, and the model is no longer exploring new paths but walking the same one again2,3. That is the waste worth removing. Switching thinking off removes it too, but removes the useful search along with it.

Thinking off: what it saves, what it costs

Both compared with the original model, thinking on.

Without thinking, the same model answers in 70 tokens instead of 739 and loses 22.3 points on the official score. The loss is very uneven: math falls from 89% to 21%, while instructions barely move and "no function fits" even improves. Any method that would push every task the same way will hurt math long before it has cut what can be cut elsewhere.

2. The challenge, the score, and my goal

The ThinkingCap test by BottleCap AI5 mixes four benchmarks, weighted equally:

Math: SVAMP6, 1,000 word problems.

Code: MBPP+7, 378 Python functions checked by unit tests.

Instructions: IFEval8, 541 requests with verifiable formatting constraints.

Tool calls: BFCL9, 840 questions: call one function correctly, pick the right one among several, or decline when none fits.

Each question is answered 8 times. The score has two numbers:

Tokens saved1 − (rmath · rcode · rinstr · rtools)¼r = trained ÷ original output tokens, per benchmark

Accuracy change¼ (Δmath + Δcode + Δinstr + Δtools)Δ = change in accuracy, in points, per benchmark

BottleCap's own objective is modest: at least 20% shorter, losing at most 0.2 points for every 1% saved. At the start I set myself a stricter goal, with three conditions:

More than 50% fewer tokens at better than −1 point on the full test (and 52% as a safety margin).

A single training procedure run from the original model, not a chain of fixes on top of fixes.

It should generalise: the shorter thinking should carry over to tasks it was never trained on.

And one way of working: explore first, exploit later. Try many methods on small, cheap screens, then put the compute into the one that works.

Every model I trained, on one chart

Each dot is one trained model on the full test. Right means shorter, up means more accurate; the blue box is my goal. Click a method to show only its models.

          Methods 

          Question type 

      More views

The grey curve shows the usual trade-off: most of my models fell close to it. So the question for any method is not just "how much does it save?" but "does it land above the curve?". Online DPO, the blue dots in the box, landed about three points of accuracy above it.

3. Where the original model's tokens go

Before changing anything, I ran the original model on the full test and read its answers. Four findings shaped everything after:

        1Loops eat tokens52.5%of all tokens sit in the 13.5% of answers that loop
        2Wrong answers are long×2.9a wrong answer is about three times longer than a right one
        3Thinking pays unevenly+68 / −7points thinking is worth on math, and on "no function fits"
        4Reasoning is repetitive2.95hesitation words ("wait", "hmm"…) per 100 words

The first two are about trimming, the last two about density.

3.1 Loops

Loop = the same 8-word sequence four or more times in the reasoning, or an answer that runs into the 4,096-token cap.

13.5% of the original's answers loop. Because they are its longest answers by far, those 13.5% of answers hold 52.5% of all its tokens. Removing loops alone would save a lot; but on tool calls almost nothing loops, so loops cannot be the whole story.

52.5% of its tokens sit in 13.5% of its answers

Left: the answers that loop hold half of all tokens. Middle: where each task spends its tokens. Right: before and after training.

After training, the share of looping answers falls from 13.5% to 2.9%, and the tokens they hold from 52.5% to 24%, mostly on code.

3.2 Long answers are wrong answers

On every task, wrong answers are two to three times longer than right ones: 1,473 tokens on average against 501. When the model does not know, it re-checks and restarts. For training, this is good news: a preference for shorter answers and a preference for right answers rarely pull in opposite directions.

Wrong answers are longer

Average length of a right and a wrong answer, original model. On the right: how many times longer the wrong one is, before and after training.

The whole distribution shifts, not just the runaway answers: the median answer goes from 354 to 162 tokens.

3.3 Where thinking pays

Where thinking pays

Questions in ten groups, from those the original thinks least about (left) to most (right). The shaded band is what thinking adds.

original, thinking onoriginal, thinking offafter training

On math, thinking is nearly everything. On instructions it is worth one point, and on "no function fits" it is harmful: the model talks itself into calling a function it should decline. These values became the recipe's per-task settings: math keeps a high minimum length and may not save too much; "no function fits" may be cut hard.

3.4 How it writes: trimmed, not condensed

The words a model uses when it reconsiders ("wait", "hmm", "alternatively", "let me double-check") are a cheap window into its behaviour. The trained model uses them less often per word it writes: it hesitates less.

The words it uses while it hesitates

How often each hesitation word appears, the single number, and what it looks like in a real answer.

Fewer paragraphs, same paragraphs

Each block is one paragraph of thinking; its width is the number of words in it.

This is the most telling result. 12 paragraphs become 4, each about the same size: the model drops whole steps; it does not write each step tighter. It writes 60% fewer words, but each paragraph is the same length and uses the same variety of words (0.73 distinct words per word inside a paragraph, before and after). It also uses fewer distinct words overall (101 → 66): it explores fewer paths, and settles on the first one that works. Compare one step of the same problem:

Original
"First, I need to figure out the difference between the number of cakes he made initially and the number he sold. The problem mentions he sold 8 cakes, and initially he had 40. So, the difference would be 40 minus 8, right? Let me check that."

After training
"First, the initial number of cakes is 40, and he sold 8. The question is asking how many more he made than sold. So, the difference would be initial − sold. That would be 40 − 8 = 32."

The step itself is written about the same way. What changed is what comes after it: the original adds a second check, a third ("but let me make sure"), and a closing one ("I don't see any other steps here"); the trained model checks once and stops.

A useful way to put a number on this is thinking density: how much accuracy each thinking token buys.

thinking density = (accuracy with thinking − accuracy without thinking) ÷ thinking tokens × 100

There are two ways to raise it:

Trim: remove thinking that adds nothing (loops, repeated checks, restarts).

Condense: say the same useful reasoning in fewer words.

The idea is close to what Trajectory calls intelligence density4. For this model, thinking density rises from 4.3 to 10.9, a 2.5× gain. Since each step is written the same way, almost all of that gain comes from trimming whole steps, not from condensing them. The model did not become a more concise writer; it became a more decisive one. That is also where its remaining errors come from (section 9).

What the cut tokens were worth

Average accuracy over all tasks against thinking tokens per answer. The shaded part is what training removed.

40%
50%
60%
70%
80%0100200300400500600700
thinking tokens per answer →

thinking off · 46.5%
after training · 259 tokens · 74.7%
original · 670 tokens · 75.5%
first 259 tokens: +28.2 pointsΔ value ≈ 10.9 points per 100 tokens
next 411 tokens (cut): +0.85 pointsΔ value ≈ 0.2 points per 100 tokens

Put differently, look at the value of the tokens that were cut. The first 259 thinking tokens buy 28.2 points of accuracy over thinking off; the next 411, the ones training removed, bought 0.85. A cut token was worth about 0.2 points per 100 tokens, twenty times less than the average thinking token. On math the cut tokens were worth 0.7 points per 100, on code 0.16, and on instructions and "no function fits" less than nothing: removing them raised accuracy.

Two measures from other papers agree:

Hesitation words10: words like "wait", "hmm" or "let me verify" fall from 2.95 to 2.45 per 100 thinking words.

Thinking after the answer3: on math, the original writes half of its thinking after the correct number first appears; the trained model, a third.

4. Watch it think

The same question, the same random draw, both models, replayed at the same speed.

        Task 
        Example 
        Next example →
        1×4×Replay

Original Qwen3-0.6B

After training

5. What I tried, and why most of it failed

This was my first machine learning project, and I treated it as one: a place to learn by trying ideas, not only to win. I chose exploring over exploiting, following my own intuitions about what should work.

Grouped by kind, the families were:

Without training: prompts and test-time tricks. Example: a system prompt asking it to think briefly.

Changing the weights directly: steering, pruning. Example: pushing its internal activations toward "brief".

Imitation (SFT): fine-tuning on short answers; cutting traces at a "settle point". Example: training it to reproduce its own shortest right answer.

Reinforcement learning: GRPO, DRPO, length rewards, a bigger model as a coach. Example: rewarding right answers, minus a penalty per token.

Preference training (DPO): per-task adapters, offline DPO, online DPO. Example: "this short right answer beats that long one".

Without training

Prompts. A system prompt asking the model to think briefly saves about 22%, at −2.7 points. Math follows the instruction; code and tool calls mostly ignore it. Switching thinking off entirely saves 94% and costs 22.3 points. A prompt changes how long the model writes on average, not its judgement of when it is done.

Changing the weights directly

Steering is different from prompting: instead of asking, it edits the model's internal activations. I computed a "brief versus verbose" direction from the model's own short and long answers and added it to the activations while it writes. It shortened answers by up to 18% at no cost on a small screen, but stronger pushes cost 5 to 8 points, and it did not make the model better at knowing when to stop.

Pruning removes the least important weights (Wanda14). At 10% removed the model behaved the same; beyond 20% it thought longer. Removing weights does not remove habits.

Imitation

Fine-tuning on short answers. For each question I kept the model's shortest correct answer out of 8 and trained it to reproduce those answers. It saved only 12% at −0.7 on my screen. The surprise came when I used the very same answers differently, as pairs ("this short right answer is better than that long one"): the saving jumped to 40–48%. Imitation shows the model a short answer; a pair shows it what to prefer, and why. That contrast pointed me to preference training.

Cutting at a "settle point". Find, for each answer, the earliest point where its final answer is already decided, cut the trace there, and train the model to stop at that point. It reached 57% saved on the full test, but at −3.5, and math paid most. Fixing the accuracy needed a second "repair" stage on top. It worked in the end, but it required an oracle to find the cuts and a separate repair step, it was hard to reproduce cleanly, and it felt like fitting the benchmarks rather than teaching the model something.

Reinforcement learning

BottleCap itself warns that reinforcement learning works poorly on models this small. I wanted to see it for myself. GRPO11 with a length penalty, DRPO12, a ShorterBetter-style reward13 and a larger model as a coach all barely shortened the original model: the best run saved 11% on my screen. This is not proof that reinforcement learning cannot work here. I tuned these runs much less than the DPO ones, and the likeliest cause is a settings problem: the length penalty was tiny next to the right-or-wrong reward, so the model had little reason to shorten. A stronger penalty risks the opposite, cutting steps the model needs, and finding the balance would have taken many more runs. Preference training needed no such balance: a pair says directly which answer is better.

Preference training

One adapter per task. I trained a separate small adapter for each task, each allowed to cut as much as its task tolerated, and merged them. On the full test the merge saved 53.6% but at −3.3 points: the adapters interfered with each other once combined.

Offline DPO15,3. Pairs built once from answers stored before training, then trained in one go: the model never learns from the new, shorter answers it starts to write. Its best model reached the goal, 54.1% at −0.93, but only at the end of a six-stage chain of rounds and repairs. Rebuilt in a clean three-step procedure from the original model, it stopped at 51.7% / −0.86: inside the box, short of the margin, and still not the single procedure I wanted.

Online DPO rebuilds the pairs every round from the model's own latest answers. Its first version, without any brake, went too far: 60.5% at −4.8, with math losing 9 points. With the brake and the other rules of section 7, it gave the best model, reproduced with a second seed.

6. Hypotheses

By the time I settled on online DPO, the analysis and the failures had turned into four working hypotheses:

Thinking density can rise. A large share of the thinking adds nothing (section 3), so accuracy per thinking token can go up a lot without new knowledge.→ Judge every model on saving and accuracy together, never on length alone.

The model already knows the short way. Among its 8 answers to a question there is usually one that reaches the right result without the extra checks. The knowledge is there; the habit of using it is not.→ Train on the model's own answers, not on a bigger model's.

More thinking would not make it better. Its long answers are mostly its wrong ones (section 3.2); adding thinking mostly adds loops.→ Prefer the shorter right answer, and the right answer over the wrong one.

Less thinking should not make it worse, as long as the cut removes repetition and not the steps a task actually needs: on instructions thinking is worth one point, on math 68 (section 3.3).→ A minimum length and a brake per task.

The strategy follows: do not teach the model a new style. Help it get to the answer it already reaches, by the shorter path it already sometimes takes.

7. The recipe: online DPO against a fixed reference

DPO16Trains on pairs: for one question, answer A is better than answer B. A becomes more likely, B less.

No reward modelNo reward to design, no sampling during the update: stable and cheap on one GPU.

Online, not offlinePairs are rebuilt every round from the current model's answers, so it learns from its latest behaviour.

Why I bet on online. Offline pairs go stale: after a few steps the model no longer writes the long answers it is being told to avoid, so the pairs stop teaching anything, and each further gain needed a new stage. Online pairs follow the model as it changes. The cost is control: a model that keeps learning from its own outputs can run away from itself. The first online runs did exactly that, and most of the recipe below exists to stop it.

FigureOne round of online DPO

1 · Fresh questions780 new questions, matched to the test by task, length and difficulty

2 · Answerthe current model answers each one 8 times

3 · Pairagainst the original's stored answers: shorter beats longer, right beats wrong, recovery

4 · Brakea task that saved enough or lost 2 points gets no "think less" pairs next round

5 · Train15 small steps, the original model as the fixed reference

7.1 Training data17

Every round uses 780 questions the model has never seen: 220 math, 140 code, 200 instructions and 220 tool calls. Each round's set is built to look like the test, not just in its mix of tasks but in how hard the questions are. I group candidate questions by how often the original model solves them (never, rarely, about half the time, usually, always) and draw them so that each task's difficulty profile and length match the test's within 3 points of solve rate.

TaskSources of training questions

MathGSM8K and MetaMathQA word problems, filtered to SVAMP-like difficulty

CodeKodCode tasks, plus generated tasks (from round 20)

InstructionsRequests from public instruction datasets (no_robots, Dolly, OpenAssistant), combined with IFEval-style verifiable constraints

Tool callsToolACE, Glaive function calling and Hermes function calling, plus generated "no function fits" and one-function questions (from round 10)

Every candidate is checked against the test for shared 8-word sequences and dropped if it overlaps. When the public pools ran out of hard tool and code questions, I generated new ones with a frontier language model and kept only verified ones.

7.2 The pairing rules

The original model's 8 stored answers to each question are the yardstick.

Shorter beats longer. A right answer at least 30% shorter than the original's right answers is preferred to a longer right answer.

Right beats wrong. Always.

Recovery. If the current model gets wrong a question the original got right, its wrong answer is paired against the original's right one.

The first rule alone caused collapse. Pushed round after round, a task kept shortening until its accuracy fell off a cliff, and once it had fallen the model could not climb back. The fix was a minimum length per task: a short answer is never preferred if it is below that task's floor. The floors come from the collapse points I had measured across all my earlier full tests. For each task I took the best saving any model reached while losing at most one or two points on that task: not a true ceiling, but a practical one.

TaskBest saving at ≤ 1 pt lostBest at ≤ 2 pts lostOnline DPO, round 16

Math47%60%61% (−2.8)

Code62%66%59% (−2.0)

Instructions48%50%47% (+0.5)

One function45%50%49% (−1.1)

Several functions51%51%50% (0.0)

No function fits66%66%57% (+5.9)

The aim was to bring each task close to its own limit, and no further. Math went slightly past its limit, which is where most of the remaining loss is.

7.3 The brake

After each round, each task is checked on that round's questions. If it has saved more than its cap, or lost 2 points of accuracy, it gets no "think less" pairs in the next round, only right-over-wrong and recovery pairs. The brake is checked again every round, so a task can resume once it is safe.

7.4 The fixed reference

DPO measures every change against a reference model. I always use the original model as the reference, never the previous round's. Every round, the model is gently pulled back toward where it started, which is what stopped the slow drift of my first online runs.

7.5 Settings and cost

LoRA18 adapter of rank 16; 15 training steps of 16 pairs per round (20 in the first round); constant learning rate 1.4 × 10⁻⁵; β = 0.1. One round takes 30 to 40 minutes on a single L4 GPU, sixteen rounds about 9 hours, and a full test about two hours.

8. Results

ModelTokens savedAccuracy (95% interval)

Online DPO, round 1054.1%−1.68

Online DPO, round 1554.0%−0.56

Online DPO, round 1654.8%−0.78 [−1.55, −0.07]

Online DPO, round 1952.5%−0.77 [−1.49, −0.04]

Second seed, round 1551.9%−0.52 [−1.25, +0.21]

Second seed, round 1652.1%−0.58 [−1.26, +0.06]

Round 16, per taskTokens savedAccuracy change

Math61%−2.8

Code59%−2.0

Instructions47%+0.5

One function49%−1.1

Several functions50%0.0

No function fits57%+5.9

A limitation: the average hides a real cost on math (−2.8) and code (−2.0), offset by a gain on "no function fits" (+5.9), where the original used to talk itself into calling a function it should have declined.

Two seeds, round by round

The second seed starts slower, about seven points of saving behind around rounds 4 to 6, and catches up by round 15. Then both runs plateau. I kept training to round 21 to see whether the policy would eventually collapse. It does not: accuracy holds and the saving gives back about two points. The brake is part of the reason: in the later rounds the three tool tasks sat at their saving caps and got no more "think less" pairs. These lines come from each round's training questions, which read accuracy a few points higher than the official test; the official numbers are in the table above.

9. Where the accuracy went

The average loss of 0.78 points hides questions that got worse and others that got better.

What training changed, beyond noise

Questions that got worse (left) or better (right) more often than chance would explain.

On math, 75 more questions got worse than chance explains19, and none got better. On "no function fits" it is the reverse: 21 more got better and 13 fewer got worse. On instructions, it is all chance.

The clearest cost is reliability. On math, the share of questions the model gets right in all 8 attempts (G-Pass@820) falls from 70% to 62%, while average math accuracy fell only 2.8 points. So the trained model now misses some questions the original always solved. This qualifies the fourth hypothesis: less thinking does not hurt when it removes repetition, but now and then the model also skips a check it needed.

The longer the question, the bigger the saving

The longer the original thinks on a question, the more the trained model saves. On math, the accuracy cost sits in the middle and long questions: the ones with several steps, where one skipped check is enough to change the number. Reading the lost answers confirms it. In most of them the trained model stopped at less than half the original's length. Loops are no longer the main cause of errors; stopping too early is.

10. Does the saving carry over to new tasks?

A model trained on questions matched to four benchmarks could simply have learned those benchmarks. To check, I ran it on four benchmarks never used or imitated in training.

Four tasks it never saw

Four benchmarks never used in training, 8 answers per question, against the original model.

The four tasks are ARC-Challenge, CommonsenseQA, MMLU and HumanEval21. The shorter thinking carries over: 52% to 63% fewer tokens on tasks it never saw, with accuracy within a few points of the original. ARC-Challenge (−3.4) is the only change clearly beyond the noise. An earlier offline model, which saved less on the official test, saved slightly more on these unseen tasks and lost less on two of them. My reading is that online DPO pushed the model further down in thinking than any earlier model, and on unfamiliar tasks that deeper cut shows up first as lost accuracy. The brake is the lever for this: tighter caps per task would trade some saving for safety on unfamiliar tasks. I have not tested that yet.

11. Why it works

The short answers were already there. Among the original's 8 answers to a question, there is usually one that reaches the right result without the extra checks. Online DPO does not have to teach a new style, only to move probability toward a behaviour the model already produces. That is a much easier thing to learn from a few hundred pairs per round, and it explains why the paragraphs themselves did not change (57 → 55 words each): the model is choosing among its own ways of writing, not inventing a new one.

Learning from its latest self keeps the signal alive. Offline pairs describe a model that no longer exists after a few steps. Online pairs always contrast what the model does now with what the original did, so every round has something left to teach, until the brake says it is enough. Offline, my clean three-step run stopped at 51.7%; online, one procedure reached 54.8%.

The fixed reference keeps it anchored. Measuring every round against the original model, both in the pairs and in the loss, turns sixteen rounds of training into sixteen small steps from the same starting point, not a random walk. Two runs with different random seeds land close together: 54.8% / −0.78 and 52.1% / −0.58.

Per-task control matches per-task value. Thinking is worth 68 points on math and minus 7 on "no function fits". Floors, caps and the brake let each task go as far as its own value allows, which is why the average lands about three points of accuracy above the curve that a single global pressure falls on (Exhibit 2).

The failures point the same way. Every method that taught length directly (imitation, cuts, length rewards, prompts) taught the model to be short, not to know when to stop. The method that compared the model's own answers taught it to choose: the same short answers saved 12% as imitation targets, and 40–48% as pairs.

Did it get smarter?

12. But shorter is not smarter

On the left, what makes a model shorter (fewer thinking tokens); on the right, what would make it smarter (more reliable, or more capable).

Shorter?Yes

✓−57%tokens per answer

✓×2.5thinking density

✓409 → 58above the cheapest right answer

✓−17%hesitation words

✓51% → 34%math thinking after the answer

Smarter?Not yet

=−0.78average accuracy

✗+75math questions worse, beyond noise

✗70 → 62%math right 8 times out of 8

✓+21“no function fits” better, beyond noise

✓+2.1thinking off: trained vs original

Two different things can make a model use fewer tokens:

Answer efficiency: the same average accuracy for fewer tokens. This is what the challenge scores, and what I achieved.

Reasoning efficiency: the same answers, question by question, for fewer tokens. That would mean the model had learned to think better, not just less.

Telling the two apart takes more than one number:

Thinking density: the accuracy thinking adds, per 100 thinking tokens. 4.3 → 10.9: much better.

Retention: the share of the original's right answers the model keeps. 93.1%: some are lost.

Rescue: the share of the original's wrong answers the model fixes. 17.9%: on average this makes up for the losses, which is why the average barely moves.

Accuracy at a fixed thinking budget: cap both models at the same number of thinking tokens. A model that truly thinks better should be more accurate at every cap. I have not run this one yet.

Two more observations:

It is close to its cheapest answer. Take, for each question, the shortest right answer either model ever wrote: the original spends 409 tokens above it on average, the trained model 58.22

The model itself changed only a little. With thinking switched off in both, the trained model answers 2.1 points better than the original. That is a hint, not proof: part of it is formatting, and I have not yet run the same check on other trained models.

Why it matters: for anyone who relies on a model, a swap is not neutral. A model that gets the same average by solving different questions is less predictable than the one it replaced. Measuring that properly is the subject of the next section.

13. Measuring how much the answers move

BottleCap states the goal of this kind of work as spending "fewer tokens getting to the answer, and change nothing else"23. The challenge's score cannot fully check the second half: it compares averages, and cannot see the swap from section 12. So I built a stricter, question-by-question test: does the model still solve every question the original solved, as reliably, with fewer tokens?

13.1 The test

One question at a time. Both models answer each question 8 times. Run the original twice and it already disagrees with itself, so a drop only counts if it is bigger than chance.

Chance, computed exactly. If the two models were the same, any split of a question's 16 answers into 8 and 8 would be equally likely. That gives, per question, the drop chance alone would produce.

Two numbers. The net change: how many questions' worth of accuracy was gained or lost. And the excess churn: how much the answers moved in either direction, beyond what chance alone produces, added up over the 2,759 questions. "73 questions" means as much movement as 73 questions going from always right to always wrong, or the reverse.

A check that it is fair. The original against a second run of itself gives an excess churn of +2 questions [−9, +11]: zero, within noise.

The closest earlier measures count losses question by question, but from a single answer and without correcting for chance19, or correct for chance approximately without giving a verdict per task24.

13.2 What it shows

70 right before, still right

5 right before, now wrong

4 wrong before, now right

21 wrong both times

        Out of every 100 answers, best model against the original. The average moves by less than one point, yet 9 answers in 100 swap. Part of that swapping is chance; what is left after correcting for it is the excess churn in the table below.

ModelTokens savedAverage accuracyNet changeExcess churn

Gentlest run42%+0.1+7 questions54 questions

Best model (round 16)55%−0.8−23 questions73 questions

Most aggressive run63%−2.1−58 questions95 questions

Original, thinking off94%−22.3−800 questions717 questions

Every model I trained moves answers beyond chance, on every task, even where its average does not change: on "several functions" the best model's net change is exactly zero, yet 12 questions' worth of answers moved each way.

The real trade-off is steeper than the average suggests. The more a model saves, the more its answers move: 54 questions at 42% saved, 73 at 55%, 95 at 63%.

Crude cuts pay more. The run that cut each answer at a fixed "settle point" saved 57% but lost a net 104 questions, four times the best model.

The saving generalizes. On four benchmarks it never trained on (ARC-Challenge, CommonsenseQA, MMLU, HumanEval), the model saves 52% to 63% with accuracy within a few points of the original; only ARC-Challenge (−3.4) drops beyond noise. So the shorter thinking is not tied to the training tasks. I have not yet run the per-question test on those benchmarks: that is the next check.

13.3 Why this is useful

For evaluating any method that shortens reasoning: report the net change and the excess churn per task next to the saving, not just the average.

For training: the same test can drive the brake (stop shortening a task once its losses go beyond chance) and the choice of the final model (the biggest saving within a loss budget). A run with stricter rules, designed from this analysis, is training now.

The honest claim for this work is answer efficiency: the model is about half as expensive to get an answer from, at nearly the same average accuracy. It does not yet think more efficiently, by this test.

14. What this means for legal AI

Most of a reasoning model's work is deliberation; only a small part reaches the reader. This project showed that much of that deliberation can go, and that the hard part is proving nothing important went with it.

That second part is the one I care about. I am training to be a lawyer, and legal work has the same shape as what I studied here: long reasoning in writing, a short output that matters. But in law, tokens are not the bottleneck. Even a long model answer costs far less than an hour of a lawyer's time. What matters is reliability: whether the answer is right, and right every time the same question is asked.

Three directions follow, and they are what I would like to work on:

Evaluating legal AI question by question. Ask the same legal question many times and measure how often the answer changes, against chance, as in section 13. A lawyer cannot rely on a model that is right on average.

Building verifiers for legal work. Everything in this project depended on telling a right answer from a wrong one thousands of times. Math has a number and code has unit tests; legal tasks need their own checks: that a citation exists and says what is claimed, that a redline matches the instruction, that a clause cross-reference is consistent.

Post-training on legal tasks. With such checks, the method in this write-up applies directly: let a model compare its own answers and learn to prefer the reliable ones.

There are early signs this works at scale: Trajectory reports4 that density-aware training on Harvey's Legal Agent Benchmark kept the same pass rate while cutting a model's output from about 90,000 to 37,000 tokens per task. The next step is to make the same gains measurable, question by question, in the terms a lawyer would accept.

If you work on legal AI evaluation or post-training, I would like to hear from you.

Credits and references

The ideas in this project are my own, but several papers shaped how I thought about them. Numbers match the notes in the text.

Qwen Team, 2025. Qwen3 Technical Report.

Muennighoff et al., 2025. s1: Simple test-time scaling (budget forcing; longer thinking helps, then flattens).

Chen et al., 2024. Do NOT Think That Much for 2+3=? On the Overthinking of o1-like LLMs (outcome efficiency; DPO and SimPO on shortest versus longest correct answers).

Yap, D. A., 2026. Intelligence Density. Trajectory, Field Notes (24 September 2026).

BottleCap AI, 2026. ThinkingCap challenge (test set, scoring rules and baseline).

Patel et al., 2021. Are NLP Models really able to Solve Simple Math Word Problems? (SVAMP).

Austin et al., 2021. Program Synthesis with Large Language Models (MBPP); Liu et al., 2023. Is Your Code Generated by ChatGPT Really Correct? (EvalPlus, MBPP+).

Zhou et al., 2023. Instruction-Following Evaluation for Large Language Models (IFEval).

Patil et al., 2024. Berkeley Function-Calling Leaderboard (BFCL).

Zhao et al., 2025. Let LLMs Break Free from Overthinking via Self-Braking Tuning (arXiv 2505.14604): overthinking markers.

Shao et al., 2024. DeepSeekMath (GRPO).

Li, Chen, Lin and Yang, 2025. DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization (ICLR 2026).

Yi, Wang and Li, 2025. ShorterBetter: Guiding Reasoning Models to Find Optimal Inference Length for Efficient Reasoning (arXiv 2504.21370).

Sun et al., 2023. A Simple and Effective Pruning Approach for Large Language Models (Wanda).

Kimi Team, 2025. Kimi k1.5: Scaling Reinforcement Learning with LLMs, in particular its "long2short" methods.

Rafailov et al., 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model.

Training questions: GSM8K (Cobbe et al., 2021), MetaMathQA (Yu et al., 2023), KodCode (Xu et al., 2025), no_robots (Hugging Face H4, 2023), databricks-dolly-15k (Databricks, 2023), OpenAssistant Conversations (Köpf et al., 2023), ToolACE (Liu et al., 2024), Glaive function calling (Glaive AI, 2023), Hermes function calling (Nous Research, 2024).

Hu et al., 2021. LoRA: Low-Rank Adaptation of Large Language Models.

Yan et al., 2021. Positive-Congruent Training (negative flip rate); Echterhoff et al., 2024. MUSCLE (flip rates for LLM updates).

Liu et al., 2024. Are Your LLMs Capable of Stable Reasoning? (G-Pass@k).

Unseen tasks: ARC-Challenge (Clark et al., 2018), CommonsenseQA (Talmor et al., 2019), MMLU (Hendrycks et al., 2021), HumanEval (Chen et al., 2021).

Pu et al., 2025. ThoughtTerminator: Benchmarking, Calibrating, and Mitigating Overthinking in Reasoning Models (global overthinking).

BottleCap AI, 23 October 2026. ThinkingCap Gemma-4-26B-A4B, company blog post: bottlecapai.com/post/thinkingcap-gemma-4-26b.

Cacioli, 2026. arXiv 2604.27405: a Reliable Change Index (Jacobson and Truax, 1991) adapted to compare two LLM versions item by item, 10 samples per item.

Draft notes: a reproducibility section (code, data, exact commands) is still to come.

  Exit full screen ✕
