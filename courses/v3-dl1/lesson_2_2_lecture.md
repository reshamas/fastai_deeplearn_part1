# Lesson 2_2

### `1:01:10` Digging in to the math
- But, let's now dig in and actually understand it more completely
- [back to number "8" represented by matrix of numbers].   So, we're going to create this mathematical function that takes the numbers that represent the pixels and spits out probabilities for each possible class.  And, by the way, a lot of the stuff that we're using here, we are stealing from other people who are awesome and so we are putting their details here.  So, like please check out their work because they've got great work that we are highlighting in our course.  
- I really like this idea of this little animated GIF of the numbers.  So, thank you to [Adam Geitgey](https://medium.com/@ageitgey) for creating that.  And I guess that was probably on Quora by the looks of this Medium.  oh yes, it was... that terrific Medium post, I remember, I've had a whole series of Medium posts. 
- `1:02:02` So, let's look and see what... how we create one of these functions.  And let's start with the simplest function I know:
y = a * x + b
- That's a line, right?  That's a line, and the gradient of the line is here (a) and the intercept of the line is here (b).  So, hopefully when we said that you need to know high school math to do this course, these are the things we're assuming you remember.  If we do kind of mention some math thing which I'm assuming you remember, and you don't remember it, don't freak out, right.  Happens to all of us.  [Khan Academy](https://www.khanacademy.org/) is actually terrific.  It's not just for school kids.  Go to Khan Academy, find the concept that you need a refresher on.  And he explains really well, so strongly recommend checking that out.  You know, remember, I'm just a philosophy student, right.  So, I... all the time I'm trying to either remind myself about something or I never learnt something and so we have the whole internet to teach us these things.
- So, I'm going to rewrite this slightly:  y = a_1 * x + a_2
- So, let's just replace `b` with `a2`.  Just give it a different name.  So, there's another way of saying the same thing.  
- and then another way of saying that would be if I could multiply a_2 by the number 1:  y = a_1 * x + a_2 * 1
- ok this still is the same thing ok.  and so now at this point I'm actually going to say let's not put the number 1 there but let's put an X1 here and an X2 here:   y = a_1 * x_1 + a_2 * x_2  ---> and I'll say x_2=1
- ok so so far this is not you know this is pretty early high school math.  this is multiplying by 1, which I think we can handle.
- ok so these two are equivalent, with a bit of renaming:
  - y = a * x + b
  - y = a_1 * x_1 + a_2 * x_2 
- now in machine learning, we don't just have one equation. we've got lots right. so if we've got some data that represents the temperature versus the number of ice creams sold, then we kind of have lots of dots.  and so each one of those dots we might hypothesize,
you know, is based on this this formula y equals a 1 X 1 plus a 2 X 2 all right.  and so basically there's lots of... so this is our Y, this is our X... there's lots of values of Y so we can stick a little "i" here.
- and there's lots of values of X so we can stick little X here:
  - y_i = a * x_i + b
  
- okay so the way we kind of do that is a lot like numpy indexing right.  But rather than things in square brackets or PyTorch indexing,  rather than things in square brackets, we kind of put them down here, in our, kind of, in the subscript of our equation
  - - y = a_i * x_i_1 + a_i * x_i_2 
  
-ok so this is now saying there's actually lots of these different Y_i's based on lots of different xi1 and xi2. ok but notice there's only this is still only one of each of these that's it so 
- these these things here called the **coefficients** or the **parameters** 
- so this is our linear equation and this is still we're going to say that every X I 2 is equal to 1 ok. 
- why did I do it that way? because I want to do linear algebra. why do I want to do in linear algebra?
- well one reason is because Rachel teaches the world's best linear algebra course. so if you're interested check out computational linear algebra for coders so it's a good opportunity for me to throw in a pitch for this course which we make no money but never mind.
- but more to the point right now it's going to make life much easier right. because I hate writing loops. I hate writing code right. I just I just want the computer to do everything for me.
- at anytime you see like these little I subscripts that sounds like you're going to have to do loops and all kind of stuff.
- but what you might remember from school is that when you've got like two things being multiplied together, two things being multiplied together, and then they get added up , that's called a **dot product**.
- and then if you do that for lots and lots of different numbers i then that's called a **matrix product**. so in fact this whole
thing can be written like this. rather than lots of different way eyes.
- we can see there's one vector called Y which is equal to one matrix called x times one vector called a.
- now at this point, I know a lot of you don't remember that so that's fine we have a picture to show you I don't know who created this
so now I do somebody called Andres touts credit this fantastic thing called matrix multiplication XYZ:  http://matrixmultiplication.xyz/
- and here we have a matrix by a vector and we're going to do a matrix vector product.  Go!  that times that times that plus plus plus plus that times that times that Plus that Plus that times that times a plus plus plus plus.  finished!  that is what matrix vector multiplication does in other words it's
just that except his version is much less messy .  

### `1:08:25` having enough data
okay so let's this is actually an excellent spot to have a little break and find out what questions we have coming through our students.  What are they asking Rachel? 
- RT:  When generating new image data set, how do you know how many images are enough?  what are ways to measure enough?
- JH:  yeah that's a great question so another possible problem you have is you don't have enough data.  How do you know if you don't have enough data?  Because you found a good learning rate:
  - because if you make it higher, then it goes off into massive losses.  
  - if you make it lower it goes really slowly 
so you've got a good learning rate.  nd then you train for such a long time that your error starts getting worse, okay.  so you
know that you're trained for long enough.  and you're still not happy with the accuracy. it's not good enough for the, you know, teddy bear cuddling level of safety you want.  so if that happens, there's a number of things you can do and we'll learn about some of them
during... e'll learn pretty much all of them during this course.  but **one of the easiest ones is get more data**.  now if you get more data, then you can train for longer, get a higher accuracy, lower error rate without overfitting.  unfortunately there's no shortcut I wish there was. I wish so somewhere to know ahead of time how much data you need. 
- but I will say this:  most of the time **you need less data than you think** so organizations very commonly spend too much time gathering data. getting more data than it turned out they actually needed.
- so get a small amount first and see how you go.

### `1:10:00` having unbalanced classes

- RT:  what do you do if you have unbalanced classessuch as 200 Grizzlies and 50 teddies ah nothing try it works a lot of people ask this question about how do I deal with unbalanced data I've done lots of analysis withunbalanced data over the last couple of years and I just can't make it not work it always works so there's a there's actually a paper that said like if you want to get it slightly better then the
best thing to do is to take that uncommon class and just make a few copies of it that's called over sampling but you're like I haven't found a situation in practice where I needed todo that I've found it always just works plain for me once you unfreeze and retrain with one cycle again if you're training loss is still lower than your validation loss likely underfitting do you retrain it unfrozen again which will technically be more than one cycle or do you redo everything with a longer epoch for the cycle hey you guys asked me that last week my answers still the same I don't know I just find if you do another cycle then it'll kind of maybe generalize a little bit better if you start again do twice as long it's kind of annoying depends how patient you are it won't make much difference you know for me personally I normally just train a few more cycles but yeah it doesn't make much difference most of the time so showing the code sample where you were creating a CNN with resin at 34 for the grizzly Teddy classifier it says this requires res not resident 34 which I find surprising I had assumed that the model created by dot save which is about 85 megabytes on disk would be able to  run without also needing a copy of resin  at 34 yeah and I understand we're going to be learning all about this shortly
[Music]
you don't there's a copy of ResNet 34 written at 34 is actually how what we call an architect we're going to be learning a lot about this it's a functional form just like this is a linear functional form it doesn't take up any room it doesn't contain anything it's just a function resident 34 is just a function it doesn't contain anything it doesn't store anything I think the confusion here is that we often use a pre trained neural net that's been learned on
imagenet in this case we don't need to
use a pre trained you're on it and
actually to entirely avoid that even
getting created you can actually pass
pre-trained equals false and that'll
ensure that nothing even gets loaded
which will save you another 0.2 seconds
I guess so yeah but we'll be learning a
lot more about this so don't worry this
is a bit unclear but the basic idea is
this this thing here is is the basically
equivalent of saying is it a line or is
it a quadratic or is it a reciprocal
this is if this is just a function this
is the resonate 34 function it's a
mathematical function it has no doesn't
take any storage it doesn't have any
numbers doesn't it be loaded as opposed
to a pre-trained model and so that's why
when we used when we did it at inference
time the thing that took space is this
bit which is where we load our
parameters which is basically saying as
we're ready to find out what are the
values of a and B we have to store those
numbers but for ResNet 34 you don't
distort two numbers you store a few
million or few tens of millions of
numbers so why did we do all this well
it's because I wanted to be able to
write it out like this and then I think
I think I would write it out like this
is that we can now do that in pi torch
with no loops single line of code and
it's also going to run faster pi torch
really doesn't like loop
right it really wants you to send it a
whole equation to do all at once which
means you really want to try and specify
things in these kind of linear algebra
ways so let's go and take a look because
what we're going to try and do then is
we're going to try and take this we're
going to call it an architecture like
that this is like the tiniest world's
tiniest neural network it's got two
parameters you know a 1 and a 2 we're
going to try and fit this architecture
to some data so let's let's jump into a
notebook and generate some dots right
and see if we can get it to fit a line
somehow and the somehow is going to be
using something called s G D what is s
GD well there's two types of SGD the
first one is where I said in Lesson one
hey you should all try building these
models and try and come up with
something cool and you guys all
experimented and found really good stuff
so that's where the s would be student
that would be student gradient descent
so that's version one of fgd version two
of SGD which is what I'm going to talk
about today is where we're going to have
a computer try lots of things and try
and come up with a really good function
and that will be called stochastic
gradient descent so the other one that
you hear a lot in the on Twitter is
stochastic grad student descent so
that's the other one for you here so
we're going to jump into lesson two SGD
and so we're going to kind of go bottom
up rather than top down we're going to
create the simplest possible model we
can which is going to be a linear model
and the first thing that we need is we
need some data and so we're going to
generate some data the data we're going
to generate looks like this so this
might represent temperature and this
rate represent number of ice creams we
sell or something like that but we're
just going to create some synthetic data
that we know is following a line and so
as we build this we're actually going to
learn a little bit about PI torch as
well so basically the way we're going to
generate this data
is by creating some coefficients a 1
will be 3 and a 2 will be 2 and we're
going to create some like which looks at
before basically a column of numbers
through axis and a whole bunch of ones
and then we're going to do this X at a
what is X at a X at a in Python means a
matrix product between X and a it
actually is even more general for that
it can be a vector vector product a
matrix vector product a vector matrix
product or a matrix matrix product and
then actually in pi torch specifically
it can mean even more general things
where we get into higher rank tensors
which we will learn all about very soon
right but this is basically the key
seeing that's going to go on in all of
our deep learning the vast majority of
the time our computers are going to be
basically doing this multiplying numbers
together at adding them up which is the
surprisingly useful thing to do
ok so we basically are going to generate
some data by by creating a line and then
we're going to add some random numbers
to it but let's go back and see how we
created X 1/8 so I mentioned that you
know we've basically got these two
coefficients three and two and you'll
see that we've wrapped it in this
function called
cancer you might have heard this word
tensor before who's heard the word
tensor before about two-thirds of you
okay so it's one of these words that
sounds scary and apparently if you're a
physicist it actually is scary but in
the world of deep learning is actually
not scary at all tensor means array okay
it means array so specifically it's an
array of a regular shape right so it's
not an array where Row one has two
things and Row three has three things
and row four has one thing what you call
a jagged array that's not a tensor a
tensor is any array which has a
rectangular or cube or whatever you know
as a shape where every element every row
is the same length
and then every column is the same length
so four by three matrix would be a
tensor a vector of length four would be
a tensor a 3d array of length three by
four by six would be a tensor
that's all intensity is okay and so we
have these all the time
for example an image is a three
dimensional tensor it's got number of
rows by number of columns by number of
channels normally red green blue so for
example a kind of a vga texture would be
640 by 480 by 3 or actually we do things
backwards so when people talk about
images they normally go width by height
but when we talk mathematically we
always go a number of rows by number of
columns so it'd actually be 480 by 6 40
by 3 that will catch you out we don't
say dimensions so with tensors we use
one of two words we had to say rank or
or axes rank specifically means how many
axes are there how many dimensions are
there so an image is generally a rank 3
tensor so what we've created here is a
rank 1 tensor or also known as the
vector right but like in math people
come up with slightly different words or
actually not they come up with very
different words for slightly different
concepts why is a one dimensional array
a vector and a two dimensional arrays
and matrix and then a three dimensional
array does that even have a name not
really doesn't have a name like it
doesn't make any sense we also you know
with computers we try to have some
simple consistent naming conventions
they're all called tensors rank 1 tensor
rank two tensor rank 3 tensor you can
certainly have a rank 4 tensor if you've
got 64 images then that would be a rank
4 tensor of 64 by 480 by 640 by 3 for
example so tensors are very simple they
just mean arrays and so
in play torch you say tensor and you
pass in some numbers and you get back in
this case just a list I get back a
vector okay so this then represents our
coefficients the slope and the intercept
of our line and so because remember
we're not actually going to have a
special case of ax plus B instead we're
going to say there's always this second
x value which is always 1 you can see it
here always 1 which allows us just to do
a simple matrix vector product ok so
that's that's a and then we wanted to
generate this X array of data which is
going to have we're going to put random
numbers in the first column and a whole
bunch of ones in the second column so to
do that we basically say 2 pi torch
create a rank two tensor actually notice
I said again we see the PI torch that we
want to create a tensor of n by 2 so
since we passed in a total of 2 things
we get a rank two tensor the number of
rows will be N and the number of columns
will be 2 and in there every single
thing in it will be a 1 that's what
torch dot ones means and then this is
really important you can index into that
just like you can index into a list in
Python but you can put a colon anywhere
and a colon means every single value on
that axis or every single value on that
dimension so this here means every
single row and then this here means
column 0 so this is every row of column
0 I want you to grab a uniform random
number and here's another very important
concept in PI torch anytime you've got a
function that ends in an underscore it
means don't return to me that uniform
random number but replay
whatever this is being called on with
the result of this function so this
takes column zero and replaces it with a
uniform random number between minus 1
and 1 so there's a lot to unpack there
right but the good news is those two
lines of code plus this one which we're
coming to cover 95% of what you need to
know about pay torch how to create an
array how to change things in an array
and how to do matrix operations on an
array okay so that's a there's a lot to
unpack but these these small number of
concepts are incredibly powerful so I
can now print out the first five rows
okay so colon 5 is standard - slicing
syntax to say the first five rows so
here are the first five rows two columns
looking like my random numbers and my
ones so now I can do a matrix product of
that X by my a add in some random
numbers to add a bit of noise and then I
can do a scatter plot and I'm not really
interested in my scatter plot in this
column of ones right there just there to
make my linear function more convenient
so I'm just going to flip plot my zero
index column against my Y's and there it
is PLT is what we universally use to
refer to the plotting library matplotlib
and that's what most people use for most
of their plotting in Python in
scientific python we use matplotlib it's
certainly a library you'll want to get
familiar with because being able to plot
things is really important there are
lots of other plotting packages lots of
them the other packages are better at
certain things than that plot lib but
like matplotlib can do everything
reasonably well sometimes it's a little
could but you know I for me I do pretty
much everything in that flight lib
because there's really nothing it can't
do even though some libraries can do
other things a little bit better or a
little bit prettier but it's really
powerful so once you know matplotlib you
can do everything so here I'm asking
matplotlib to give me a scatterplot with
my X's against my Y's and there it is
okay so this is my my dummy data
representing like you know of
temperature and ice cream sales so now
what we're going to do is we're going to
pretend we were given this data and we
don't know that the values of our
coefficients are 3 & 2 so we're going to
pretend that we never knew that we have
to figure them out okay so how would we
figure them out how would we draw a line
to fit to this data and why would that
even be interesting well we're going to
look at more about why it's interesting
in just a moment but the basic idea is
this if we can find this is going to be
kind of perhaps really surprising but if
we can find a way to find those two
parameters to fit that line to those how
many points were there and was a hundred
if we can find a way to fit that line to
those 100 points we can also fit these
arbitrary functions that convert from
pixel values to probabilities
it'll turn out that this techniques that
we that we're going to learn to find
these two numbers works equally well for
the 50 million numbers in resident 34 so
we're actually going to use an almost
identical approach so that this and this
is a bit that I found in previous
classes people have the most trouble
digesting like I often find even after
week four or week type five people will
come up to me and say I don't get it how
do we actually train these models and
I'll say it's it's SGD it's that it's
that thing we throw in the notebook with
the T numbers it's like yep it but we're
fitting a neural network so I know and
we can't print the 50 million numbers
anymore
but it is literally identically doing
the same thing and the reason this is
hard to digest is that the human brain
has a lot of trouble conceptualizing of
what an equation was fifteen milk 50
million numbers looks like and can do so
you're just kind of for now we'll have
to take my word for it that can do
things like recognize teddy deaths and
all these functions turn out to be very
powerful now we're going to learn a
little bit more in just a moment about
how to make them extra powerful but for
now this thing we're going to learn to
fit these two numbers is the same thing
that we've just been using to fit 50
million numbers okay so we want to find
what pi torch calls parameters or in
statistics you'll often hear called
coefficients these values a 1 and a 2 we
want to find these parameters such that
the line that they create minimizes the
error between that line and the points
so in other words you know if we created
you know if the if the a 1 and a 2 we
came up with resulted in this line then
we'd look and we'd see like how far away
is that line from each point I would say
oh that's quite a long way and so maybe
there was some other a 1 or a 2 which
resulted in this line and they would say
like oh how far away is each of those
points and then eventually we come up
with blue we come up with this line and
it's like Oh in this case each of those
is actually very close all right so you
can see how in each case we can say how
far away is the line at each spot away
from its point and then we can take the
average of all those and that's called
the loss and that is the value of our
loss right so you need some mathematical
function that can basically say how far
away is this line from those points
for this kind of problem which is called
a regression problem a problem where
your dependent variable is continuous so
rather than being Grizzly's or Teddy's
it's like some number between minus 1
and 6 this is called a regression
problem and for regression the most
common loss function is called mean
squared error which pretty much
everybody calls MSE you may also see our
MSE just root mean squared error and so
the mean squared error is a loss it's
the difference between some prediction
that you've made okay
which you know is like the value of the
line and the actual number of ice cream
sales and so in in the mathematics of
this people normally refer to the actual
they normally call it Y and the
prediction they normally call it y hat
as in they they write it like that and
so what I try to do like when we're
writing something like it you know means
grid error equation
there's no point writing ice cream here
and temperature here because we wanted
to apply it to anything so we tend to
use these like mathematical placeholders
so the value of mean squared error is
simply the difference between those two
squared all right and then we can take
the mean because remember that is
actually a vector or what we now call it
a rank 1 tensor and that is actually a
rank 1 tensor so it's the value of the
number of ice cream sales at each place
and so when we subtract 1 vector from
another vector we're going to be
learning a lot more about this but it
does something called element wise
arithmetic in other words it subtracts
each each one from each other and so we
end up with a vector of differences and
then if we take the square of that it
squares everything in that vector and so
then we can take the mean of that to
find the average square of the
differences between the actuals and the
predictors so
if you're more comfortable with
mathematical notation what we just wrote
there was the some of which we rounded
we do it y hat minus y squared over n
right so that equation is the same as
that equation so one of the things I'll
note here is I don't think this is you
know more complicated or unwieldy than
this right but the benefit of this is
you can experiment with it like once you
have to find it
you can use it you can send things into
it and get stuff out of it and see how
it works alright so for me most of the
time I prefer to explain things with
code rather than with math right because
I can actually they're the same now
they're doing in this case at least in
all the cases we'll look at they exactly
the same they're just different
notations for the same thing but one of
the notations is executable it's
something that you can experiment with
and one of them is abstract so that's
why I'm generally going to show code so
the good news is if you're a coder with
not much of a math background actually
you do have a math background because
code is math right if you've got more of
a math background and less of a code
background then actually a lot of the
stuff that you learned from math is
going to translate very directly into
code and now you can start to experiment
really with your math okay so this is
some lost function this is something
that tells us how good our line is so
now we have to kind of come up with what
is the line that fits through here
remember we don't know we're going to
pretend we don't know so what you
actually have to do is you have to guess
you actually have to come up with a
guess what are the values of a 1 and a 2
so let's say we guess that a 1 and a 2
are both 1 so this is our tensor a is 1
comma 1
so here is how we create that tenser and
I wanted to write it this way because
you'll see this all the time like
written out it should be 1.0 olives so
it's also it was telling of minus 1
minus 1 written out fully it would be
minus 1.0 1.0 like that's that's written
out fully we can't write it without the
point because that's now an INT not a
floating point so that's going to spit
the dummy if you try to do calculations
with that neural Nets
ok I'm lazy I'm far too lazy to type dot
0 every time Playford knows perfectly
well that if you added dot next to any
of these numbers then the whole thing is
now floats right so that's that's why
you'll often see it written this way
particularly by lazy people let me okay
so a is a chancer you can see it's
floating-point you see like even pi
torch is lazy they just put a dot they
don't bother with a zero right but if
you want to actually see exactly what it
is you can write dot type and you can
see it's a float tensor okay and so now
we can calculate our predictions with
this like random guess X at a matrix
product of X and a and we can now
calculate the mean squared error of our
predictions and their actuals and that's
our loss okay so for this regression our
loss is 0.9 and so we can now plot a
scatter plot of X against Y and we can
plot the scatter plot of X against Y hat
our predictions and there they are
okay so this is the 1 1 comma minus 1
line so it minus 1 comma 1 line and
here's actuals so that's not great - not
surprising it's just a guess so FGD or
gradient descent more generally and
anybody who's done in d engineering or
probably computer science at school will
have done plenty of this like Newton's
method what
all the stuff that you did University if
you didn't don't worry we're going to
learn it now it's basically about taking
this guess and trying to make it a
little bit better so how do we make it a
little bit better well there's only two
numbers right and the two numbers are
and the two numbers are the intercept of
that orange line and the gradient of
that orange line so what we're going to
do with gradient descent is we're going
to simply say what if we change those
two numbers a little bit what if we made
the intercept a little bit higher or a
little bit lower what if we made the
gradient a little bit more positive or a
little bit more negative so there's like
four possibility and then we can just
calculate the loss for each of those
four possibilities and see what see what
work did lifting it up or down make it
better there tilting it more positive or
more negative make it better and then
all we do is we say okay well whichever
one of those made it better that's what
we're going to do and that's it right
but here's the cool thing
for those of you that remember calculus
you don't actually have to move it up
and down and round about you can
actually calculate the derivative the
derivative is the thing that tells you
we're moving it up or down make it
better or would rotating it this way or
that way make it better
okay so the good news is if you didn't
do calculus or you don't remember
calculus I just told you everything you
need to know about it right which is
that it tells you how changing one thing
changes the function right that's what
that's what the derivative is kind of
not quite strictly speaking right close
enough also called the gradient okay so
the gradient or the derivative tells you
how changing a one up or down would
change our MSE now changing a true up or
down will change your MSE and this does
it more quickly does it more quickly
than actually moving it up and down okay
so um
in school unfortunately they forced us
to sit there and calculate these
derivatives by hand we have computers
computers can do that for us we are not
going to calculate them by hand instead
we're going to call dot bread on our
computer that will calculate the
gradient for us so here's what we're
going to do we're going to create a loop
we're going to loop through 100 times
and we're going to call a function
called update that function is going to
calculate Y hat our prediction it is
going to calculate loss now means grad
error from time to time it will print
that out so we can see how we're going
it will then calculate the gradient and
in pi torch calculating the gradient is
done by using a method called backward
so you'll see something really
interesting which is mean squared error
was just a simple standard mathematical
function pi torch for us keeps track of
how it was calculated and lets us
calculate the derivatives so if you do a
mathematical operation on a tensor in pi
torch you can call backward to calculate
the derivative what happens to that
derivative that gets stuck inside an
attribute called dot Brad so I'm going
to take my coefficients a and I am going
to subtract from them my gradient and
there's an underscore here why because
that's going to do it in place so it's
going to actually update those
coefficients a to subtract the gradients
from them right so why do we subtract
well because the gradient tells us if I
move the whole thing downwards the loss
goes up if I move the whole thing
upwards the loss goes down so I want to
like do the opposite of the thing that
makes it go up right so because our last
we want to loss to be small so that's
why we have to subtract
and then there's something here called
LR LR is our learning rate and so
literally all it is is the thing that we
multiply by the gradient why is there
any LR at all let me show you why let's
take a really simple example a quadratic
okay and let's see your algorithms job
was to find where that quadratic was at
its lowest point and so well how could
it do this well just like what we're
doing now the starting point would just
be to pick some x value at random and
then pop up here to find out what the
value of y is okay that's the starting
point and so then it can calculate the
gradient and the gradient is simply the
slope but it tells you moving in which
direction is going to make you go down
and so the gradient tells you you have
to go this way so if the gradient was
really big you might jump this way a
very long way so you might jump all the
way over to here maybe even here right
and so if you jumped over to there then
that's actually not going to be very
helpful because then you see well where
does that take us to Oh it's now worse
right we jumped too far so we want don't
want to jump too far so maybe we should
just jump a little bit maybe to here and
the good news is that is actually a
little bit closer and so then we'll just
do another little jump see what the
gradient is into another liberal jump
that takes us to here and another little
jump that takes us to here here yeah
right so in other words we find our
gradient to tell us kind of what
direction to go and like
we have to go a long way or not too far
but then we multiply it by some number
less than one so we don't jump too far
and so hopefully at this point this
might be reminding you of something
which is what happened when our learning
rate was too high so do you see why that
happened now our learning rate was too
high
meant that we jumped all the way past
the right answer further than we started
with
and it got worse and worse and worse so
that's what a learning rate to high does
on the other hand if our learning rate
is too low then you just take tiny
little steps and so eventually you're
going to get there but you're doing lots
and lots of calculations along the way
so you really want to find something
where it's kind of either like this or
maybe it's kind of a little bit
backwards and forwards maybe it's kind
of like this something like that you
know you want something that kind of
gets in there quickly but not so quickly
it jumps out and diverges not so slowly
that it takes lots of steps so that's
why we need a good learning rate and so
that's all it does so if you look inside
the source code of any deep learning
library you will find this you will find
something that says coefficients dot
subtract learning rate times gradient
and we'll learn about some minor up not
minor what about so easy bad important
optimizations we can do to make this go
faster but that's basically it there's a
couple of other little minor issues that
we don't need to talk about now one
involving zeroing out the gradients and
other involving making sure that you
turn gradient calculation off when you
do the SGD update if you're interested
we can discuss them on the forum or you
can do our introduction to machine
learning course which covers the other
mechanics of this in more detail
but this is the basic idea so if we run
update 100 times printing out the loss
from time to time you can see it starts
at 8.9 it goes down down down down down
down down and so we can then print out
scatter plots and there it is that's it
but leave it or not that's gradient
descent so we just need to start with a
function that's a bit more complex than
X at a but as long as we have a function
that can represent things like is this a
teddy bear we now have a way to fit it
okay and so let's now take a look at
this as a picture as an animation and
this is one of the nice things that you
can do with this is one of the nice
things that you can do with matplotlib
is you can take Eddie plot and turn it
into an animation mat and so you can now
actually see it updating each step so
let's see what we did here we simply
said as before create a scatter plot but
then rather than having a loop we used
matplotlib func animation so call 100
times this function and this function
just called that update that we created
earlier and then updated the Y data in
our line and so did that 100 times
waiting 20 milliseconds after each one
and there it is right so you might think
that like visualizing your algorithms
with animations is some amazing and
complex thing to do but actually now you
know it's 1 2 3 4 5 6 7 8 9 10 11 lines
of code okay so I think that is pretty
damn cool
so that is SGD visualized and so we
can't visualize as conveniently what
updating 50 million parameters in a
resonant 34 looks like but basically
doing the same thing okay and so
studying these simple versions is
actually a great way to get an intuition
so you should try running this No
book with a really big learning rate
with a really small learning rate and
see what this animation looks like that
and try get a feel for it maybe you can
even try a 3d plot I haven't tried that
yet but I'm sure it would work fine 

- `1:48:07` so the only difference between stochastic gradient descent and this is something  called mini-batches
- you'll see what we did here was we calculated the value of the loss on the whole data set on every iteration but if your data set is one and a half million images in image net that's going to be really slow right just to do a single update of your parameters you've got to
calculate the loss on one and a half million images you wouldn't want to do that so what we do is we grab 64 images or so at a time at random and we calculate the loss on those 64 images and we update our weights and then we have another 64 random images we update the weights so in other words the loop basically looks exactly the same but at this point here so it'd basically be Y square bracket and some random indexes here you know and some random indexes here and we'd basically do the same thing and well actually so it would be there right so some random indexes on our X and some random indexes on our way to do a mini batch at a time and that would be the basic difference and so once you add those you know grab a random few points each time those random few points accord your mini batch and that approach is called SGD for stochastic gradient descent okay so there's quite a bit of vocab we've just covered right so let's just remind
ourselves the learning rate is a thing that we multiply our gradient by to decide how much to update the weights by an epoch is one complete run through all of our data points all of our images so for the non stochastic gradient descent

- we just did every single loop we did the entire data set but if you've got a data set with a thousand images and your mini batch size is 100 then it would take you ten iterations to see every image once. so that would be one epoch.  the epochsare important because if you do lots of epochs then you're looking at your images lots of times and so every time you see an image there's a bigger chance of overfitting so we generally don't want to do too many epochs a mini batch  is just a random bunch of points that you use to update your weights SGD is just gradient descent using mini-batches architecture and model kind of mean the same thing in this case our architecture
is y equals XA and the architecture is  the mathematical function that you're fitting the parameters to and we're going to learn later today or next week what the mathematical function of things like ResNet 34 actually is but it's basically pretty much what you've just seen it's a bunch of matrix products parameters also known as coefficients also known as weights the set the numbers that you're updating and then loss function is the thing that's telling you how far away or how close you are to the correct answer any questions all right so these model these predictors these teddybear classifiers are functions that take pixel values and return probabilities they start with some functional form like y equals XA and they fit the parameters a using SGD to try and do the best to calculate your predictions so far we've learned how to do regression which is a single number next week we'll learn how to do the same thing for classification where we have
multiple numbers the same in the process we had to do some math we had to do some linear algebra and we had to do some calculus and a lot of people get a bit scared at that point and tell us I am NOT a math person if that is you that's totally okay but you're wrong you are a math person in fact it turns out that when in the actual academic research around this there are not math people and non math people it turns out to be entirely a result of culture and expectations so you should check out Rachel's talk there's no such thing as not a math person where she will introduce you to some of that academic research and so if you think of yourself as not a math person you should watch this so that you learn that you're wrong that your thoughts are actually there because somebody has told you you're not a math
person but there's actually no academic research to suggest that there is such a thing in fact there are some cultures like Romania and China where the not a math person concept never even appeared there that it's almost unheard of in some cultures for somebody to say I'm not a math person because they're just never entered that cultural identity so don't freak out if words like derivative and gradient and matrix product are things that you're kind of scared of it's something you can learn it's something you'll be okay with okay so the last thing that we're going to close with today oh I just got a message from Simon Willison ah Simon's telling me he's actually not that special lots of people won medals so that's the worst part about Simon is not only is he really smart he's also really modest which I think it's just awful I mean if you're going to be that smart at least be a horrible human being and you know make it okay okay so the last thing I want to close with is the idea of and we're going to look at this more next week underfitting over and over fitting we just fit a line to our data but imagine that our data wasn't actually line shaped right and so if we try to fit something which was like constant plus constant times X ie align to it that it's never going to fit very well right no matter how much we change these two coefficients it's never going to get really close on the other hand we could fit some much bigger equation so in this case it's a higher degree polynomial with lots of lots of Wiggly bits like so right but if we did that it's very unlikely we go and look at some other place to find out the temperature that it is and how much ice cream they're selling and that will get a good result because like the Wiggles are far too Wiggly so this is called overfitting we're looking for some mathematical function that fits just right to stay with a teddy bear analogies so you might think if you have a statistics background the way to make things fit just right is to have exactly the right number of parameters it's to use a mathematical function that doesn't have too many parameters in it it turns out that actually completely not the right way to think about it there are other ways to make sure that we don't over fit and in general this is called regularization regularization or all the techniques to make sure that when we train our model that it's going to work not not only well on the data its seen but on the data it hasn't seen yet so the most important thing to know when you've trained a model is actually how well does it work on data that it hasn't been trained with and so as we're going to learn a lot about next week that's why we have this thing called a validation set so what happens with the validation set is that we do our mini batch F GED training loop with one set of data with one set of teddy bears Grizzlies black bears and then when we're done we check the lost function and the accuracy to see how good is it on a bunch of images which were not included in the training and so if we do that then if we have something which is too Wiggly it'll tell us oh your loss function in your air is really bad because on the Bears that it hasn't been trained with the wiggly bits are in the wrong spot where if it was under fitting it would also tell us that your validation sets really bad so like even for people that don't go through this course and don't learn about the details of deep learning like if you've got managers or colleagues or whatever at work who are kind of wanting to like moan about AI the only thing that you really need to be teaching them is about the idea of a validation set because that's the thing they can then use to figure out you know if somebody's telling them snake oil or not, you know.  they're like hold back some data and then they get told like oh here's a model that we're going to roll out.  
- and then you say okay fine I'm just going to check it on this held out data to see whether it generalizes. there's a lot of details to get right when you design your validation set. we will talk about them briefly next week but a more full version would be in Rachel's piece on the first day. a blog called how and why to create a good validation set and this is also one of the things we go into in a lot of detail in the intro to machine learning course. so we're going to try and give you enough to get by for this course but it is certainly something that's worth deeper study as well. any
questions or comments before we wrap up?  okay good all right. well thanks everybody I hope you have a great time building your web applications. see you next week you
