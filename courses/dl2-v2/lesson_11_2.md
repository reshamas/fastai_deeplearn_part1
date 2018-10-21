# Lesson 11_2:  Neural Translation

(09-Apr-2018, live)  

- [Wiki Lesson 11](http://forums.fast.ai/t/part-2-lesson-11-wiki/14699)
- [Video Lesson 11](https://www.youtube.com/watch?v=tY0n9OT5_nA&feature=youtu.be) 
  - video length:  2:15:57
- http://course.fast.ai/lessons/lesson11.html
- Notebook:  
   * [translate.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl2/translate.ipynb)
   
## `01:13:10` After Break
- So one question that came up during the break is that some of the tokens that are missing in fast text like had a curly quote rather than a straight quote, for example.  And the question was, would it help to normalize punctuation?  And, the answer for this particular case is, probably yes, the difference between curly quotes and straight quotes is really semantic.  You do have to be very careful, though, because like it may turn out that people using beautiful curly quotes are like using more formal language and they're actually writing in a different way so I generally... you know, if you're going to do some kind of pre-processing like punctuation normalization, you should definitely check your results with and without because like *nearly always* that kind of pre-processing makes things worse even when I'm sure it won't.

### `01:14:10`  Question
- Person x (Yannet?):  Hello, what might be some ways of realizing? these sequence to sequence models besides dropout and weight decay?
- JH:  Let me think about that during the week. Yeah, it's like you know, AWD LSTM, which we've been relying on a lot, has so many great.. I mean it's all dropout, well not all dropout,  there's dropout of many different kinds. And then there's the... we haven't talked about it very much, but there's also a kind of regularization based on activations and stuff like that as well.  And on changes, and whatever.  I just haven't seen anybody put anything like that amount of work into regularization of sequence to sequence models and I think there's a huge opportunity for somebody to do like the AWD LSTM of seq to seq, which might be as simple as dealing with all the ideas from AWD LSTM and using them directly in seq to seq.  That would be pretty easy to try, I think.  

#### Steven Merity paper
- And there's been an interesting paper that actually Steven Merity's added in the last couple of weeks where he used an idea which I don't know if he stole it from me but it was certainly something I had also recently done and talked about on Twitter.  Either way, I am thrilled that he's done it which was to take all of those different AWD LSTM hyperparameters and train a bunch of different models and then use a random forest to find out with feature importance, which ones actually matter the most, and then figure out how to set them. Yes, so I think you could totally, you know, use this approach, to figure out, you know, for sequence to sequence regularization approaches, which one is the best and optimize them.  And that would be amazing. Yeah, but at the moment, I think, you know, I don't know that there are additional ideas to sequence to sequence regularization that I can think of beyond what's in that paper for regular language model stuff and probably all those same approaches would work.

## `01:16:30` Trick #1:  Bi-directional
- Okay, so tricks.  Trick #1, go **bi-directional**.  So, for classification, my approach to bi-directional that I suggested you use is take all of your token sequences, spin them around and train a new language model and train a new classifier. And I also mentioned the wiki text pre-trained model if you replace FWD with VWD in the name, you'll get the pre-trained backward model I created for you, okay, so you can use that.  Get a set of predictions and then average the predictions just like a normal ensemble, okay.  And that's kind of how we do bi-dira [bi-directional] for that classification.  There may be ways to do it, end-to-end, but I haven't quite figured them out yet.  They're not in fastai yet, and I don't think anybody's written a paper about them yet, so if you figure it out, that's an interesting line of research.  But, because we're not doing, you know, massive documents where we have kind of to chunk it into separate bits and then pool over them and whatever, we can do bi-dir very easily, in this case.  Which is literally as simple as adding `bidirectional=True` to our encoder (`self.gru_enc`).  People tend not to do bi-directional for the decoder.  I think, partly because it's kind of considered cheating, but I don't know.  Like I was just talking to somebody at the break about it.  Maybe it can work in some situations.  Although it might need to be more of an ensembling approach in the decoder because you kind of, it's a bit less obvious.  Anyway, in the encoder, it's very, very simple:  `bidirectional=True` and we now have... With `bidirectional=True`, rather than just having an RNN which is going this direction [right], we have a second RNN that's going in this direction [left].  And so that second RNN literally is visiting them each token in the opposing order.  So, when we get the final hidden state, it's here [left], rather than here [right].  But, the hidden state is of the same size, so the final result is that we end up with a tensor that's got an extra too-long axis, right.  And, depending on what library you use, often that will then be combined with the number of layers things.  So, if you've got 2 layers, and bi-directional, that tensor dimensions is now length 4.  With PyTorch, it kind of depends which bit of the process you're looking at as to whether you get a separate result for each layer and offer each bidirectional bit, and so forth.  You have to look up the docs and it will tell you inputs/outputs, tensor sizes appropriate for the number of layers and whether you have `bidirectional=True`.
- In this particular case, you'll basically see all the changes I've had to make.  So, for example, you'll see when I added bidirectional=True, my linear layer now needs `nh*2` (number of hidden times 2) to reflect the fact that we have that second direction in our hidden state now.  You'll see in `def initHidden`, it's now  `self.nl*2` now here, okay. So, you'll see there's a few places where there's been an extra two that has to be thrown in.  Yes, Yannet?
- [Yannet Interian](https://www.linkedin.com/in/interian/): Why making a decoder by the original is considered cheating?  
- JH:  Well, it's not just cheating, it's like we have this loop going on, you know.  It's not as simple as just kind of having two tensors.  And, then, like how do you turn those two separate loops into a final result.  You know, after talking about it during the break, I've kind of gone from, like, "hey, everyone knows it doesn't work" to "oh, maybe it kind of could work but it requires more thought.  It's quite possible during the week I realize it's a dumb idea and I was being stupid, but we'll think about it.
- Yannet:  Another question people had, why do you need to have an in? to that loop?
- JH:  Why do I have a what to the loop?
- YI:  Why do you need to, like have a, an end to that loop.  You have like a range.  If your range...
- JH:  range? oh, yeah, I mean it's because when I start training, everything's random.  So, this will probably never be true:  `if (dec_inp==1).all(): break`. So, later on, it'll pretty much always break out eventually.  But, yeah, it's basically like we're going to go for it.  It's really important to remember like when you're designing an architecture that when you start, the model knows nothing about anything.  So you kind of want to make sure it's doing something at least vaguely sensible.
- So, bidirectional means we had, you know, let's see how we got here, we got out to 3.58 cross-entropy loss.  With a single direction.  With bidirection, gets down to 3.51. So that improved it a bit.  That's good and as I say, it's the only... it shouldn't really slow things down too much.  You know, bidirectional does mean there's a little bit more sequential processing have to happen.  But, you know, generally it's a good win.  In the google translation model of the 8 layers, only the first layer is bidirectional because it allows it do more in parallel. So, if you create really deep models, you may need to think about which ones are bidirectional.  Otherwise, we have performance issues. okay, so 3.51 [cross-entropy loss].

## `01:22:35` Trick #2:  Teacher Forcing
- Now, let's talk about teacher forcing.
- So, teacher forcing is... I'm going to come back to this idea that when the model starts learning, it knows nothing about nothing. So, when the model starts learning, it is not going to spit out "Er" at this point [Reference:  slide on translating "He loved to eat."], it's going to spit out some random meaningless word. Because it doesn't know anything about German or about English or about the idea of language or anything.  And then it's going to feed it down here as an input and be totally unhelpful.  And so that means that early learning is going to be very, very difficult because it's feeding in an input that's stupid into a model that knows nothing.  And somehow it says, get better, right.  So, that's... it's not asking too much... it eventually gets there, but it's definitely not as helpful as we can be.  So, what if, instead of feeding in, what if instead of feeding in the thing I predicted, just now right, what if instead we feed in the actual correct word it was meant to be?  Now, we can't do that at inference time because by definition we don't know the correct word. It has to translate it.  We can't require the correct translation in order to do translation.  So, the way I've set this up is, I've got this thing called `pr_force` which is probability of forcing, and if some random number is less than that probability, then I'm going to replace my decoder input with the actual correct thing, right.  And if we've already gone too far, if it's already longer than the target sentence, `if i>=len(y): break`, I'm just going to stop.  Cause obviously I can't give it the correct thing.
- `01:24:25` So you can see how beautiful PyTorch is for this, right, because if you try to do this with some static graph thing like classic tensorflow...well, I tried, right.  Like one of the key reasons that we switched to PyTorch at this exact point in last year's class was because Jeremy tried to implement it ? forcing it in Keras and Tensorflow and went even more insane than he started.  *It was weeks of getting nowhere!*  And then I... literally on twitter, I think it was [Andrej Karpathy](https://twitter.com/karpathy) I saw announced and said something about "oh, there's this thing called PyTorch just came out and it's really cool" and I tried it that day.  By the next day, I had teaching forcing working.  And I was like "oh my gosh", you know, and all the stuff about debugging things.  It was suddenly so much easier.  And this kind of, you know dynamic stuff, is so much easier.  So this is a great example of like, "hey I get to use random numbers and if-statements" and stuff. 
- `01:25:30` So, here's the basic idea is: at the start of training, let's set `pr_force` really high, right, so that nearly always that gets the actual correct, you know, previous word.  And so, it has a useful input.  And then as I train it a bit more, let's decrease `pr_force` so that by the end, `pr_force` is zero and it has to learn properly.  Which is fine because it's now actually feeding in sensible inputs most of the time anyway.
- `01:26:04` So, let's now write something such that in the training loop, it gradually decreases `pr_force`.  So how do you do that?  Well, one approach would be to write our own training loop.  But, let's not do that because we already have a training that has progress bars and uses exponential weighted averages to smooth out the losses and keeps track of metrics.  And you know, it does a bunch of things which... they're not rocket science, but they're kind of convenient and they also kind of keep track of, you know, calling the reset for RNNs at the start of an epoch to make sure that the hidden states set to zeroes, and you know, little things like that.  We would rather not have to write that from scratch.  So what we've tended to find is that as I start to kind of write some new thing.  And I'm like, "oh, I need to kind of replace some part of the code", I then kind of add some little hook so we can all use that hook to make things easier.  In this particular case, there's a hook that I've ended up using all the damn time now.  Which is the hook called the "stepper" --> `class Seq2SeqStepper(Stepper):`. 
- `01:27:10` So, if you look at our code, `model.py`, is where our `fit` function lives.  And so the `fit` function and `model.py` is kind of, we've seen it before I think.  It's like the lowest level thing that doesn't require a "Learner".  It doesn't really require anything much at all.  It just requires a standard PyTorch model and a model data object.  You just need to know how many epochs, our standard PyTorch optimizer (`opt`), and the standard PyTorch loss function (`crit`).  
```python
def fit(model, data, epochs, opt, crit, metrics=None, callbacks=None, stepper=Stepper, **kwargs):
```
- [model.py](https://github.com/fastai/fastai/blob/87e5b32f4826238c795f6fc8a9fac381048c110b/old/fastai/model.py)
- `01:27:40` Right, so you can call... I don't... we've hardly ever used it in the class.  We normally call `learn.fit`, but `learn.fit` calls this (`def fit`), so this is our lowest level thing.  But, we filtered the source code here sometimes.  We've seen how it loops through each epoch and that loops through each thing in our batch:
```python
    for epoch in tnrange(tot_epochs, desc='Epoch'):
        if phase >= len(n_epochs): break #Sometimes cumulated errors make this append.
        model_stepper.reset(True)
        cur_data = data[phase]
        if hasattr(cur_data, 'trn_sampler'): cur_data.trn_sampler.set_epoch(epoch)
        if hasattr(cur_data, 'val_sampler'): cur_data.val_sampler.set_epoch(epoch)
        num_batch = len(cur_data.trn_dl)
        t = tqdm(iter(cur_data.trn_dl), leave=False, total=num_batch, miniters=0)
        if all_val: val_iter = IterBatch(cur_data.val_dl)
        
        for (*x,y) in t:
            batch_num += 1
            for cb in callbacks: cb.on_batch_begin()
            loss = model_stepper.step(V(x),V(y), epoch)
            avg_loss = avg_loss * avg_mom + loss * (1-avg_mom)
            debias_loss = avg_loss / (1 - avg_mom**batch_num)
            t.set_postfix(loss=debias_loss, refresh=False)
            stop=False
            los = debias_loss if not all_val else [debias_loss] + validate_next(model_stepper,metrics, val_iter)
            for cb in callbacks: stop = stop or cb.on_batch_end(los)
            if stop: return
            if batch_num >= cnt_phases[phase]:
                for cb in callbacks: cb.on_phase_end()
                phase += 1
                if phase >= len(n_epochs):
                    t.close()
                    break
                for cb in callbacks: cb.on_phase_begin()
                if isinstance(opt, LayerOptimizer): model_stepper.opt = opt.opt
                if cur_data != data[phase]:
                    t.close()
                    break
        
```
- and calls `stepper.Step`.  And so `stepper.Step` is the thing that's responsible for calling the model, getting the loss, finding the loss function and calling the optimizer.  And so, by default, `stepper.Step` uses a particular class called `class Stepper():` which, there's a few things you don't know where ?, but basically it calls the model.  So the model ends up inside `m`. Zero is the gradient, `self.opt.zero_grad()`, calls the loss function `loss = raw_loss = self.crit(output, y)`, calls backwards `loss.backward()`, does gradient clippping if necessary `if self.clip:` and then calls the optimizer `self.opt.step()`
- So, you know, they're the basic steps that back when we looked at kind of PyTorch from scratch, we had to do.  So the nice thing is, we can replace *that* with something else, rather than replacing the training loop.   So, if you inherit from `Stepper` (`class Stepper()`) and then write your own version of `step` (`def step`), you can just copy and paste the contents of step and add whatever you like.  Or if it's something that you're going to do before or afterwards, you could even call `super.step`.  In this case, I rather suspect I've been unnecessarily complicated here (`class Seq2SeqStepper(Stepper)`).  I probably could have replaced... commented out all that `01:29:20` and just said  `super().step(xs, y, epoch)` because I think this is an exact copy of everything, right.  But, you know, as I say, when I'm prototyping, I don't think carefully about how to minimize my code.  I copied and pasted the contents of the code from `step` and I added a single line to the top (`self.m.pr_force = (10-epoch)*0.1 if epoch<10 else 0`) which was to replace `pr_force` in my module with something that gradually decreased linearly for the first 10 epochs and after 10 epochs, it was 0.
- `01:30:00` So, total hack, but good enough to try it out and so the nice thing what... is that I can now, you know, everything else is the same.  I've replaced... I've added these 3 lines of code (`if (y is not None)`...) to my module

```python

class Seq2SeqRNN_TeacherForcing(nn.Module):
    def __init__(self, vecs_enc, itos_enc, em_sz_enc, vecs_dec, itos_dec, em_sz_dec, nh, out_sl, nl=2):
        super().__init__()
        self.emb_enc = create_emb(vecs_enc, itos_enc, em_sz_enc)
        self.nl,self.nh,self.out_sl = nl,nh,out_sl
        self.gru_enc = nn.GRU(em_sz_enc, nh, num_layers=nl, dropout=0.25)
        self.out_enc = nn.Linear(nh, em_sz_dec, bias=False)
        self.emb_dec = create_emb(vecs_dec, itos_dec, em_sz_dec)
        self.gru_dec = nn.GRU(em_sz_dec, em_sz_dec, num_layers=nl, dropout=0.1)
        self.emb_enc_drop = nn.Dropout(0.15)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(em_sz_dec, len(itos_dec))
        self.out.weight.data = self.emb_dec.weight.data
        self.pr_force = 1.
        
    def forward(self, inp, y=None):
        sl,bs = inp.size()
        h = self.initHidden(bs)
        emb = self.emb_enc_drop(self.emb_enc(inp))
        enc_out, h = self.gru_enc(emb, h)
        h = self.out_enc(h)

        dec_inp = V(torch.zeros(bs).long())
        res = []
        for i in range(self.out_sl):
            emb = self.emb_dec(dec_inp).unsqueeze(0)
            outp, h = self.gru_dec(emb, h)
            outp = self.out(self.out_drop(outp[0]))
            res.append(outp)
            dec_inp = V(outp.data.max(1)[1])
            if (dec_inp==1).all(): break
            if (y is not None) and (random.random()<self.pr_force):
                if i>=len(y): break
                dec_inp = y[i]
        return torch.stack(res)
```        
- `01:30:17` And the only thing I need to do other that's differently is when I call `fit` is I pass in my customized `Stepper` class, okay.  
- And so that's going to do teacher forcing.  And so we don't have bidirectional so we're just changing one thing at a time, so we should compare this to our unidirectional results which was 3.58 (val_loss), and this was 3.49.  So that was an improvement so that's great.  I needed to make sure at least 10 epochs because before that, it was cheating by using the teaching forcing, so yeah, okay.
- So, that's good.  That's an improvement.  

## `01:31:00` Trick #3:  Attention Model
- So, we've got another trick and this next trick is a... it's a bigger trick.  It's a pretty cool trick.  And it's called attention.  And the basic idea of attention is this, which is: expecting the entirety of the sentence to be summarized into this single hidden vector, `s`, is asking a lot, you know.  It has to know what was said and how it was said and everything necessary to create the sentence in general.  
- And so the idea of attention is basically like maybe we're asking too much, all right.  Particularly because we could use this form of model [reference to slide:  Predicting chars 2 to n using chars 1 to n-1, `01:31:54`], where we output every step of the loop to not just have a hidden state at the end, but to have a hidden state after every single word.  And like why not try and use that information?  It's like already there, but so far, we've just been throwing it away.
- And not only that, but bidirectional we've got every step, we've got two, you know, vectors of state that we can use.
- So, how could we use this [slide: encoder] piece of state, this piece of state, this piece of state, this piece of state and this piece of state [encoder blocks: He, loved, to, eat, .] , rather than just the **final state**?  And so the basic idea is well, let's say I'm doing this word ["liebte"], translating this word right now.  Which of these 5 pieces of state [He loved to eat .] do I want?  And, of course, the answer is:  If I'm doing... Well, actually, let's pick a more interesting word.  Let's pick this one [softmax: Er "liebte" zu essen.].  So, if I'm trying to do "loved", then clearly the hidden state I want is this one [from Embed: He "loved" to eat], right.  Because this is the word.  
- And then, for this preposition here ["zu"] (is it a preposition?, whatever).  This little word ["zu" = his] here... No, it's not a preposition.  I guess it's part of the verb.  [note from Reshama: zu=his, possessive noun].  For this part of the verb, I probably would need like this and this and this [to, loved, He] to kind of make sure that I've got kind of the tense right and know that I actually need this part of the verb and so forth.  
- So depending on which bit I'm translating, I'm going to need one or more bits of this, of these various hidden states.  And in fact, you know, like I probably want some weighting of them.  So like what I'm doing here, I probably mainly want this state right but maybe I want a little bit of that one and a little bit of that one, right.  So, in other words, for these 5 pieces of hidden state, we want a **weighted average**, right.  And we want it weighted by something that can figure out which bits of the sentence is the most important right now.  So, how do we figure out something like which bits of the sentence are the most important right now.  
  - So, how do we figure out something like which bits of the sentence are important right now?  
    - **We create a neural net.**  And we train the neural net to figure it out.  
  - When do we train that neural net?  
    - End to end
- So, let's now train two neural nets.  Well, we've actually already kind of got a bunch, right:
  - We've got an RNN encoder.
  - We've got an RNN decoder.
  - We've got a couple of linear layers.  
- What the hell?  Let's add another neural net into the mix, okay.  And this neural net is going to spit out a weight for every one of these things.  And we've got to take the weighted average at every step and it's just another set of parameters that we learn all at the same time. 
