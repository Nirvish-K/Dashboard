## **Task**

The task given was to make a proof-of-concept "platform" to optimize Meta Ads performance. Due to other schedules, I was only able to do some parts of the case study for instance I wasn't able to implement the API integration code, although I believe that would have been to show that I have the knowledge of using API's to fetch data. I currently don't have either a facebook or shopify account so no data would be in there to analyze, but in the architecture diagram section I have detailed on this process and more. The web dashboard uses sample datasets found on kaggle, go through the multiple pages of what might be included on the dashboard. Realistically I wouldn't use streamlit to create a dashboard but is it makes creating things easier to create and visualize.

## **Target Audience & Understanding**

The target audience for this platform is most likely businesses looking to optimize how they run their ads or use the performance data and analytics to make decisions, generate reports, and so on. The bonus features mentioned are predictive budgeting, A/B test automation, Ad spend ROI tracking, and email or slack notifications. So we can assume that with the approval of the user, the platform is able to run ad campaigns and access budgeting data (in other words connects to the bank and has a budget allocated to perform A/B test automation). The platform might also scale up to include data from shopify and facebook so creating a robust framework will help speed things later on.

## **Architecture Diagram**

In the frontend we use react and javascript, using rest API's we can connect to the API gateway which will handle things like authorization, rate limiting, or error handling. Authorization is done through OAuth 2.0 which is a safe way to log in users and also allows us to generate tokens (& refresh tokens) with which we can fetch data from facebook (meta) and shopify API”s. We also set up cron jobs or could use apache airflow to schedule periodic data fetching (gain access through tokens stored in database) for analysis. Finally we store the data, which includes things like user activity/preferences and the generated ML/AI data into NoSQL databases such as mongoDB (NoSQL better for later scalability). The architecture doesn’t dive deep into the data pipeline but we want to be able to process/transform the data to make it usable for analysis. We can do this using some tools like Apache spark, and for data visualization we can embed tableau into the frontend. The architecture also doesn’t get into the details of the payment and notification system which are separated as they are not directly related to the data modeling. It also doesn’t suggest using a microservices architecture since that's something that’s suited for scaling up, for startups it’s best to use a monolithic architecture. But considering the platform intended to design it’s best to keep in mind scalability from the start.

Security:
The token stored in the databases need to be encrypted,, furthermore if a payment system was to be designed then we want user to sign up and link their facebook (meta) and shopify accounts instead. We will stored the passwords by salting and running a hashing function to keep them safe.

## **Scalability**

When scaling up we might want to ingest real-time data into making predictions or add more sources of data and to manage such things we might consider using tools like Apache Kafka. Also for scaling up we can consider switching to a cloud infrastructure such as AWS which provides many services like security, storage, compute, analytics and more which will help serve a large number of users and save costs. Consider the platform tasked to design if a business wants many of their employees to access the same data you need to set up RBAC and need authentication services which are provided by AWS. The cloud computing platform also offers notification services which can be incorporated into the platform to notify users of performance delays.

## **Edge Cases & Errors**

The API gateway will handing rate limiting and if there is no data to be computed then nothing can be generated. However, the platform could consider adding a page which display recommendations and insights compiled from analyzing the data from all users of the platform given they agree to sharing data (something like popular trends in marketing).
