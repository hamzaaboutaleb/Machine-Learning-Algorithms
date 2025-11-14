### statistic and probability :
---
- What is Marginal probability ? <br>
it is simply the chance of one specific event happening without worrying about what happens with other events . for example if you re looking at the probability of it raining tomorrow , you only care about the chance of rain not what happens with other weather canditions like wind or temperature .

- What are the probability Axioms ? <br>
are just basic rules that help us undestand how probabilities work . there are three main ones : 
1. Non negativity axiom : Probability cant be negative , always between 0 and 1 
2.  Normalization axiom : if something is certain to happen , its probability is one 
3. Additivity axiom : if two events cant happen at the same time (like rolling a 3 or a 4 on a die) the chance of eather one happening is the sum of their individual chances 

- whats the difference between dependent and independent events in probability ? 
independent events : two events are independent if one event doesnt change the likelihood of the other happening . for example , flipping a coin twice - the first flip doesnt affect the second flip so the proba of both events happening is jujt the product of their individual probabilities Dependent events : if one affect the likelihood of the other event. if u draw a card from a deck and dont put it back , the chance of a second card depends on what the first card was. 

- What is canditional probability ? 
refers to the probability of an event occuring given that an event already occured. Mathematically , it is defined as the probability of event A occuring , giving that event B has occured and is denoted P(A|B) . the formula for canditional probability : 
P(A∣B)=P(A∩B)​/P(B)

- what is Bayes Theorem and when we use it in data science ? 
___
Bayes Theorem helps us figure out the probability of an event happening based on some prior knowkedge or evidence.  Its like updating our guess about something when we learn new things , the formula for Bayes Theorem is : 
P(A∣B)=P(B∣A)⋅P(A)​/P(B)

- Define variance and canditional variance : 
___
variance is a way to measure how spread out or different the numbers in a dataset are from the __average__, the variance is low. If the numbers are all close to the average , the variance is low , if the numbers are spread far apart , the variance is high.<br>
canditional variance is similar but it looks at how much a variable changes when we know something else about is. For example, imagine you want to know how much people's height varies based on their age. The conditional variance would tell you how much height changes for a specific age group, using the knowledge of age to focus on the variability within that group.