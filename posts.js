const posts = [
    {
        "id": "post-1769942400",
        "title": "Types of Machine Learning Algorithms",
        "author": "Saravana Kumar",
        "date": "2026-02-01",
        "tags": [
            "machine-learning",
            "basics",
            "algorithms"
        ],
        "image": "https://images.unsplash.com/photo-1496307042754-b4aa456c4a2d?q=80&w=1600&auto=format&fit=crop&crop=entropy",
        "placeholder": "https://images.unsplash.com/photo-1496307042754-b4aa456c4a2d?q=10&w=40&auto=format&fit=crop",
        "excerpt": "Learn the differences between supervised unsupervised and reinforcement learning.",
        "content": "<h2>Understanding Machine Learning Algorithms</h2>\n<p>In the realm of artificial intelligence, machine learning algorithms play a pivotal role in enabling systems to learn from data without being explicitly programmed. This blog post aims to shed light on three primary types: supervised learning, unsupervised learning, and reinforcement learning.</p>\n\n<h3>Supervised Learning</h3>\n<p>Supervised learning is a technique where the algorithm learns from labeled data, i.e., data that has been categorized or classified in advance. It's like teaching a child to recognize animals by showing them pictures and labeling each one.</p>\n<ul>\n<li>Example: Sentiment analysis (classifying text as positive, negative, or neutral).</li>\n<li>Tip: Use it when you have access to labeled data and want your model to make accurate predictions based on that information.</li>\n\n<h3>Unsupervised Learning</h3>\n<p>In contrast, unsupervised learning deals with unlabeled data. The algorithm must find patterns and relationships within the data on its own. It's like letting a child play with building blocks to figure out the patterns by themselves.</p>\n<ul>\n<li>Example: Clustering (grouping similar data points together).</li>\n<li>Tip: Use it when you want your model to identify hidden structures or relationships within your data.</li>\n\n<h3>Reinforcement Learning</h3>\n<p>Reinforcement learning is a method where an agent learns to make decisions based on reward feedback from its environment. It's like training a pet using rewards and punishments to shape its behavior.</p>\n<ul>\n<li>Example: Training an autonomous vehicle to navigate a city efficiently (the vehicle receives rewards for reaching the destination quickly and safely).</li>\n<li>Tip: Use it when you want your model to learn optimal strategies in complex, dynamic environments.</li>\n\n<h2>Conclusion</h2>\n<p>Each machine learning algorithm has its unique strengths and applications. By understanding their differences, we can make informed decisions about which one to use for specific problems. As technology advances, the potential for these algorithms to revolutionize industries continues to grow.</p>\n<em>Stay tuned for more insights into the world of machine learning!</em>"
    },
    {
        "id": "post-1769938567",
        "title": "Introduction to Machine Learning",
        "author": "Saravana Kumar",
        "date": "2026-02-01",
        "tags": [
            "machine-learning",
            "basics",
            "ai"
        ],
        "image": "https://images.unsplash.com/photo-1496307042754-b4aa456c4a2d?q=80&w=1600&auto=format&fit=crop&crop=entropy",
        "placeholder": "https://images.unsplash.com/photo-1496307042754-b4aa456c4a2d?q=10&w=40&auto=format&fit=crop&crop=entropy",
        "excerpt": "Understand what machine learning is and how it differs from traditional programming.",
        "content": "<p>Delve into the fascinating world of Machine Learning, a revolutionary approach to data analysis that's transforming industries worldwide.</p>\n\n<h2><strong>Understanding Machine Learning</strong></h2>\n<p>Machine Learning (ML) is a subset of Artificial Intelligence (AI) that empowers computer systems to automatically learn and improve from experience without being explicitly programmed. It's all about creating algorithms capable of recognizing patterns in data and using them to make informed decisions or predictions.</p>\n\n<h3><em>How does Machine Learning differ from traditional programming?</em></h3>\n<p>Traditional programming requires explicit instructions for a computer to perform specific tasks. On the other hand, Machine Learning allows computers to learn patterns and make decisions based on these patterns without being explicitly programmed.</p>\n\n<h2><strong>Types of Machine Learning</strong></h2>\n<p>Machine Learning can be broadly categorized into three types: Supervised Learning, Unsupervised Learning, and Reinforcement Learning. Each has its unique approach to data analysis and problem-solving.</p>\n\n<h3><em>Supervised Learning:</em></h3>\n<p>In supervised learning, the algorithm learns from labeled data, meaning the correct answer is provided during training. This type of learning is ideal for tasks such as image recognition and speech recognition.</p>\n\n<h3><em>Unsupervised Learning:</em></h3>\n<p>Unsupervised learning involves dealing with unlabeled data, where the algorithm must find patterns without any guidance about the correct answers. Clustering and dimensionality reduction are common examples of this type of learning.</p>\n\n<h2><strong>Practical Applications</strong></h2>\n<p>Machine Learning has found numerous applications across various sectors, from recommending products to predicting stock market trends. For instance, Netflix uses ML for personalized movie recommendations, while Amazon leverages it for product suggestions based on browsing and purchasing history.</p>\n\n<h2><strong>The Future of Machine Learning</strong></h2>\n<p>As data continues to grow exponentially, the demand for machine learning will rise, driving advancements in AI and transforming our world in ways we can hardly imagine today. Whether it's self-driving cars or personalized healthcare, ML is set to play a pivotal role in shaping our future.</p>\n\n<p>Embark on this journey of understanding Machine Learning, and unlock the potential to revolutionize your field of work and beyond.</p>"
    },
    {
        "id": "post-1769876929",
        "title": "Getting Started with Docker",
        "author": "Saravana Kumar",
        "date": "2026-01-31",
        "tags": [
            "docker",
            "devops",
            "containers"
        ],
        "image": "https://images.unsplash.com/photo-1496307042754-b4aa456c4a2d?q=80&w=1600&auto=format&fit=crop&crop=entropy",
        "placeholder": "https://images.unsplash.com/photo-1496307042754-b4aa456c4a2d?q=10&w=40&auto=format&fit=crop&crop=entropy",
        "excerpt": "Learn how to use Docker to containerize your applications and simplify deployment.",
        "content": "<p>Embark on a transformative journey and master the art of application containerization with Docker! This blog post serves as your comprehensive guide, outlining essential steps to simplify deployment processes.</p>\n\n<h2><strong>Understanding Docker</strong></h2>\n<p>Docker is an open-source platform that automates the deployment, scaling, and management of applications. It enables developers to package their applications with all necessary dependencies into lightweight, portable containers.</p>\n<h3><em>Key Benefits:</em></h3>\n<ul>\n<li>Isolation: Containers share the host operating system's kernel, minimizing resource duplication and enhancing performance</li>\n<li>Portability: Applications can run consistently across various environments, be it development, testing, or production</li>\n<li>Efficiency: Docker images reduce the size of application packages, making them faster to download and deploy</li>\n</ul>\n\n<h2><strong>Getting Started with Docker</strong></h2>\n<p>To begin, install Docker on your preferred operating system. Follow the instructions provided on the official <a href=\"https://docs.docker.com/get-docker/\">Docker documentation page</a>.</p>\n<h3><em>First Command:</em></h3>\n<p>Verify successful installation by executing the command: <code>docker run hello-world</code>. This command downloads and runs the \"hello-world\" Docker image, displaying a simple greeting upon completion.</p>\n\n<h2><strong>Building Your First Docker Image</strong></h2>\n<p>Create a new directory for your application and navigate to it using the terminal. In this example, we will build a Dockerfile that creates a basic Node.js application with NPM dependencies.</p>\n<pre><code># Dockerfile\nFROM node:14 # Use official Node.js 14 image as base\nWORKDIR /app\nCOPY package*.json ./\nRUN npm install\nCOPY . .\nEXPOSE 3000\nCMD [ \"npm\", \"start\" ]\n</code></pre>\n<h3><em>Build and Run Your Docker Image:</em></h3>\n<p>To build the image, execute <code>docker build -t my-node-app .</code>, replacing \"my-node-app\" with your desired container name. Once built, run the image using <code>docker run -p 3000:3000 my-node-app</code></p>\n\n<h2><strong>Conclusion</strong></h2>\n<p>Docker offers a powerful solution for developers and DevOps professionals seeking to streamline application deployment. By containerizing your applications, you can guarantee consistency across environments, reduce resource consumption, and improve the overall development workflow.</p>\n<p>Ready to dive deeper into Docker? Explore the official <a href=\"https://docs.docker.com/get-started/\">Docker documentation</a> for more tutorials, guides, and best practices.</p>\n<hr />\n<p>We hope this guide has been informative and helpful in your Docker journey! Stay tuned for future blog posts on advanced topics and practical case studies.</p>"
    },
    {
  id:'post-6',
  title:'Bagging vs Boosting Explained: Core Ideas, Differences, and Interview Tips',
  author:'Saravana Kumar',
  date:'2025-09-22',
  tags:['machine-learning','ensemble-learning','bagging','boosting','interview-prep'],
  image:'https://images.unsplash.com/photo-1555949963-aa79dcee981c?q=80&w=1600&auto=format&fit=crop&crop=entropy',
  placeholder:'https://images.unsplash.com/photo-1555949963-aa79dcee981c?q=10&w=40&auto=format&fit=crop&crop=entropy',
  excerpt:'Bagging and Boosting are powerful ensemble techniques. Learn how they work, when to use each, and how to explain them clearly in interviews.',
  content:`
    <p>Bagging and Boosting are ensemble learning techniques used to improve model performance. While both combine multiple models, their goals and approaches are fundamentally different.</p>

    <h3>1. What is Bagging (Bootstrap Aggregating)?</h3>

    <p><strong>Core idea:</strong> Train multiple independent models in parallel on different random samples of the data and aggregate their predictions.</p>

    <h4>How it Works</h4>
    <ul>
      <li>Create multiple datasets using bootstrap sampling (random sampling with replacement).</li>
      <li>Train the same model on each dataset.</li>
      <li>Combine results:
        <ul>
          <li>Classification → Majority voting</li>
          <li>Regression → Averaging</li>
        </ul>
      </li>
    </ul>

    <h4>Goal</h4>
    <p>Reduce variance. Bagging is especially effective for unstable models that tend to overfit.</p>

    <h4>Intuition</h4>
    <p>“Let many weak opinions vote — noise cancels out.”</p>

    <h4>Best For</h4>
    <ul>
      <li>High variance models</li>
      <li>Overfitting problems</li>
    </ul>

    <h4>Example</h4>
    <p><strong>Random Forest</strong> — a classic bagging-based algorithm.</p>

    <h4>Pros</h4>
    <ul>
      <li>Reduces overfitting</li>
      <li>Easy to parallelize</li>
      <li>Stable predictions</li>
    </ul>

    <h4>Cons</h4>
    <ul>
      <li>Does not significantly reduce bias</li>
      <li>Requires many models → higher computation cost</li>
    </ul>
    <h3>2. What is Boosting?</h3>

    <p><strong>Core idea:</strong> Train models sequentially, where each new model focuses more on correcting the mistakes of the previous ones.</p>

    <h4>How it Works</h4>
    <ul>
      <li>Train a base model.</li>
      <li>Identify misclassified or high-error data points.</li>
      <li>Increase their importance (weights).</li>
      <li>Train the next model focusing more on these errors.</li>
      <li>Combine all models using weighted voting or weighted sum.</li>
    </ul>

    <h4>Goal</h4>
    <p>Reduce bias (and variance in many cases).</p>

    <h4>Intuition</h4>
    <p>“Learn from mistakes, step by step.”</p>

    <h4>Best For</h4>
    <ul>
      <li>Weak learners</li>
      <li>Complex patterns</li>
      <li>High bias models</li>
    </ul>

    <h4>Examples</h4>
    <ul>
      <li>AdaBoost</li>
      <li>Gradient Boosting</li>
      <li>XGBoost, LightGBM, CatBoost</li>
    </ul>

    <h4>Pros</h4>
    <ul>
      <li>Very high accuracy</li>
      <li>Handles complex relationships well</li>
    </ul>

    <h4>Cons</h4>
    <ul>
      <li>Sensitive to noise and outliers</li>
      <li>Sequential training → slower</li>
      <li>Can overfit if not regularized</li>
    </ul>

    <h3>3. Bagging vs Boosting (Interview Comparison)</h3>

    <table border="1" cellpadding="5" cellspacing="0">
      <thead>
        <tr>
          <th>Aspect</th>
          <th>Bagging</th>
          <th>Boosting</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Training</td>
          <td>Parallel</td>
          <td>Sequential</td>
        </tr>
        <tr>
          <td>Focus</td>
          <td>Reduce variance</td>
          <td>Reduce bias</td>
        </tr>
        <tr>
          <td>Data sampling</td>
          <td>Random with replacement</td>
          <td>Weighted (focus on errors)</td>
        </tr>
        <tr>
          <td>Model dependency</td>
          <td>Independent</td>
          <td>Dependent</td>
        </tr>
        <tr>
          <td>Noise sensitivity</td>
          <td>Low</td>
          <td>High</td>
        </tr>
        <tr>
          <td>Overfitting control</td>
          <td>Strong</td>
          <td>Needs tuning</td>
        </tr>
        <tr>
          <td>Example</td>
          <td>Random Forest</td>
          <td>XGBoost</td>
        </tr>
      </tbody>
    </table>

    <h3>4. When to Use Which?</h3>

    <p><strong>Use Bagging when:</strong></p>
    <ul>
      <li>Your model is overfitting</li>
      <li>The dataset is noisy</li>
      <li>You want stable predictions</li>
      <li><strong>Example:</strong> Decision Trees</li>
    </ul>

    <p><strong>Use Boosting when:</strong></p>
    <ul>
      <li>Your model is underfitting</li>
      <li>You need higher accuracy</li>
      <li>Data is relatively clean</li>
      <li>Complex relationships exist</li>
    </ul>

    <h3>5. One-Liner Interview Answers</h3>

    <p><strong>Bagging:</strong><br/>
    “Bagging reduces variance by training multiple independent models on bootstrapped datasets and aggregating their predictions.”</p>

    <p><strong>Boosting:</strong><br/>
    “Boosting reduces bias by training models sequentially, where each model focuses on correcting the mistakes of the previous ones.”</p>

    <p><strong>Final takeaway:</strong> Bagging brings stability, Boosting brings accuracy. Choosing the right one depends on whether your problem suffers more from variance or bias.</p>
  `
}
,

//  <div class="inpost-img" data-placeholder="https://images.unsplash.com/photo-1555949963-aa79dcee981c?q=10&w=80&auto=format&fit=crop&crop=entropy">
//       <img data-src="https://images.unsplash.com/photo-1555949963-aa79dcee981c?q=80&w=1200&auto=format&fit=crop&crop=entropy" alt="Bagging vs Boosting visualization">
//     </div>


    {
  id:'post-5',
  title:'SOLID Principles in Low Level Design: What, When, How (Interview-Friendly Guide)',
  author:'Saravana Kumar',
  date:'2025-09-20',
  tags:['solid','lld','oop','system-design','interview-prep'],
  image:'https://images.unsplash.com/photo-1517433456452-f9633a875f6f?q=80&w=1600&auto=format&fit=crop&crop=entropy',
  placeholder:'https://images.unsplash.com/photo-1517433456452-f9633a875f6f?q=10&w=40&auto=format&fit=crop&crop=entropy',
  excerpt:'SOLID principles help you design clean, scalable systems—but they should not be forced. Learn what each principle means, when to use it, and how to apply it naturally in LLD interviews.',
  content:`
    <p>SOLID principles are one of the most discussed topics in Low Level Design (LLD) interviews. However, many developers struggle with <em>when</em> to use them and whether all of them are really necessary.</p>

    <p>This guide explains SOLID in a practical, interview-oriented way—focusing on <strong>what to apply, when to apply, and how to avoid over-engineering</strong>.</p>

    <h3>What SOLID Really Means in LLD</h3>
    <p>SOLID is not a checklist to be applied blindly. It is a set of design signals that indicate whether your code is easy to change, extend, and maintain.</p>
    <p>In interviews, evaluators care more about your <strong>design reasoning</strong> than strict rule-following.</p>

    <h3>S – Single Responsibility Principle (SRP)</h3>
    <p>A class should have only one reason to change. This does not mean one method—it means one responsibility.</p>
    <p><strong>When to use:</strong> Always. SRP is the foundation of good LLD.</p>
    <p><strong>How to apply:</strong> Ask, “Who will request a change in this class?” If multiple stakeholders can request changes, split the class.</p>
    <p><strong>Example:</strong> Separate order creation, payment processing, and notification logic into different services.</p>

    <h3>O – Open/Closed Principle (OCP)</h3>
    <p>Classes should be open for extension but closed for modification.</p>
    <p><strong>When to use:</strong> When new features or variations are expected.</p>
    <p><strong>How to apply:</strong> Identify change points and introduce interfaces or strategies.</p>
    <p><strong>Example:</strong> Use a <code>PaymentStrategy</code> interface to add new payment methods without changing existing code.</p>

    <h3>L – Liskov Substitution Principle (LSP)</h3>
    <p>Subclasses should be usable wherever their parent class is expected, without breaking behavior.</p>
    <p><strong>When to use:</strong> Whenever inheritance is involved.</p>
    <p><strong>How to apply:</strong> Validate that child classes do not weaken or disable parent behavior.</p>
    <p><strong>Red flag:</strong> Overridden methods throwing exceptions or changing expected behavior.</p>
    <p>If LSP is violated, prefer composition over inheritance.</p>

    <h3>I – Interface Segregation Principle (ISP)</h3>
    <p>Clients should not be forced to depend on methods they do not use.</p>
    <p><strong>When to use:</strong> When interfaces become large or serve multiple unrelated clients.</p>
    <p><strong>How to apply:</strong> Split interfaces based on behavior, not entities.</p>
    <p><strong>Example:</strong> Separate <code>Drivable</code>, <code>Flyable</code>, and <code>Sailable</code> interfaces instead of one large <code>Vehicle</code> interface.</p>

    <h3>D – Dependency Inversion Principle (DIP)</h3>
    <p>High-level modules should depend on abstractions, not concrete implementations.</p>
    <p><strong>When to use:</strong> Always at system boundaries such as databases, external services, and APIs.</p>
    <p><strong>How to apply:</strong> Use interfaces and inject dependencies via constructors.</p>
    <p><strong>Benefit:</strong> Improves testability and reduces coupling.</p>

    <h3>Is It Necessary to Use All SOLID Principles?</h3>
    <p>No. Applying all SOLID principles everywhere often leads to over-engineering.</p>

    <ul>
      <li><strong>SRP:</strong> Always recommended</li>
      <li><strong>OCP:</strong> Apply only when change is expected</li>
      <li><strong>LSP:</strong> Relevant only when using inheritance</li>
      <li><strong>ISP:</strong> Useful for large or growing interfaces</li>
      <li><strong>DIP:</strong> Essential at boundaries</li>
    </ul>

    <h3>How to Achieve SOLID Naturally in LLD Interviews</h3>
    <p>Instead of forcing principles, follow this design flow:</p>
    <ol>
      <li>Identify entities</li>
      <li>Separate responsibilities (SRP)</li>
      <li>Identify change points (OCP)</li>
      <li>Validate inheritance (LSP)</li>
      <li>Break large interfaces (ISP)</li>
      <li>Inject dependencies (DIP)</li>
    </ol>

    <p><strong>Key takeaway:</strong> You don’t apply SOLID directly. You design cleanly, and SOLID naturally emerges.</p>

    <p><strong>Interview-ready line:</strong><br/>
    “I don’t force SOLID upfront. I apply principles only when they reduce coupling or support future changes.”</p>
  `
}
,

//  <div class="inpost-img" data-placeholder="https://images.unsplash.com/photo-1517433456452-f9633a875f6f?q=10&w=80&auto=format&fit=crop&crop=entropy">
//       <img data-src="https://images.unsplash.com/photo-1517433456452-f9633a875f6f?q=80&w=1200&auto=format&fit=crop&crop=entropy" alt="SOLID principles overview">
//     </div>

    {
  id:'post-4',
  title:'How to Achieve SOLID Principles in Low Level Design (Step-by-Step)',
  author:'Saravana Kumar',
  date:'2025-09-18',
  tags:['solid','lld','system-design','oop','interview-prep'],
  image:'https://images.unsplash.com/photo-1517694712202-14dd9538aa97?q=80&w=1600&auto=format&fit=crop&crop=entropy',
  placeholder:'https://images.unsplash.com/photo-1517694712202-14dd9538aa97?q=10&w=40&auto=format&fit=crop&crop=entropy',
  excerpt:'SOLID is not applied upfront—it emerges through the right LLD steps. Learn how to apply each principle naturally during design.',
  content:`
    <p>SOLID principles are often misunderstood as rules to apply at the end of design. In reality, they naturally emerge when you follow a structured Low Level Design (LLD) approach.</p>

    <h3>Step 1: Identify Entities</h3>
    <p>Start by extracting core nouns from the problem statement. These become your primary classes.</p>
    <p><strong>Example:</strong> Order, Item, Payment, Invoice, User</p>
    <p>This step lays the foundation for clean object modeling.</p>

    <h3>Step 2: Separate Responsibilities (SRP)</h3>
    <p>Each class should have only one reason to change. If a class handles multiple concerns, split it.</p>
    <p><strong>Example:</strong> Separate <code>OrderService</code>, <code>PaymentService</code>, and <code>NotificationService</code>.</p>
    <p><strong>Interview tip:</strong> Ask yourself, “Who will request this change?”</p>

    

    <h3>Step 3: Look for Change Points (OCP)</h3>
    <p>Identify areas where new features are likely to be added. Design those parts for extension, not modification.</p>
    <p><strong>Example:</strong> Use a <code>PaymentStrategy</code> interface to add UPI, Card, or Wallet without changing existing code.</p>

    <h3>Step 4: Validate Inheritance (LSP)</h3>
    <p>Ensure child classes can fully replace parent classes without breaking behavior.</p>
    <p><strong>Red flag:</strong> Overridden methods that throw exceptions or reduce functionality.</p>
    <p>If LSP is violated, prefer composition instead.</p>

    <h3>Step 5: Break Large Interfaces (ISP)</h3>
    <p>Clients should not be forced to depend on methods they do not use.</p>
    <p><strong>Example:</strong> Split a large <code>Vehicle</code> interface into <code>Drivable</code>, <code>Flyable</code>, and <code>Sailable</code>.</p>

    <h3>Step 6: Inject Dependencies (DIP)</h3>
    <p>High-level modules should depend on abstractions, not concrete implementations.</p>
    <p><strong>Example:</strong> Inject <code>PaymentGateway</code> instead of instantiating <code>StripePayment</code> directly.</p>

    <p><strong>Key insight:</strong> You don’t “apply SOLID” explicitly. You follow good LLD steps, and SOLID naturally falls into place.</p>

    <p><strong>Interview takeaway:</strong> Explain <em>why</em> a principle is used, not just <em>what</em> it is.</p>
  `
}
,

// <div class="inpost-img" data-placeholder="https://images.unsplash.com/photo-1517694712202-14dd9538aa97?q=10&w=80&auto=format&fit=crop&crop=entropy">
    //   <img data-src="https://images.unsplash.com/photo-1517694712202-14dd9538aa97?q=80&w=1200&auto=format&fit=crop&crop=entropy" alt="SOLID principles LLD flow">
    // </div>
{
  id:'post-2',
  title:'OOPS Concepts Explained: Writing Clean, Scalable Object-Oriented Code',
  author:'Saravana Kumar',
  date:'2025-09-15',
  tags:['oops','object-oriented','java','python','interview-prep'],
  image:'https://images.unsplash.com/photo-1518779578993-ec3579fee39f?q=80&w=1600&auto=format&fit=crop&crop=entropy',
  placeholder:'https://images.unsplash.com/photo-1518779578993-ec3579fee39f?q=10&w=40&auto=format&fit=crop&crop=entropy',
  excerpt:'OOPS is the foundation of Low Level Design. Understand abstraction, encapsulation, inheritance, and polymorphism with practical examples.',
  content:`
    <p>Object-Oriented Programming (OOPS) is a way of designing software around real-world entities. It helps in building systems that are modular, reusable, and easy to maintain.</p>

    <h3>1. Abstraction</h3>
    <p>Abstraction means exposing only what is necessary and hiding internal details. It focuses on <em>what</em> an object does, not <em>how</em> it does it.</p>
    <p><strong>Example:</strong> A <code>PaymentService</code> exposes <code>pay()</code> without revealing whether it uses UPI, Card, or Wallet internally.</p>

    <h3>2. Encapsulation</h3>
    <p>Encapsulation is bundling data and methods together and restricting direct access to the internal state of an object.</p>
    <p><strong>Example:</strong> Private fields with public getters/setters ensure controlled access to class data.</p>

   

    <h3>3. Inheritance</h3>
    <p>Inheritance allows a class to reuse properties and behavior of another class. It represents an <em>is-a</em> relationship.</p>
    <p><strong>Example:</strong> <code>Car</code> and <code>Bike</code> inheriting from a <code>Vehicle</code> base class.</p>

    <h3>4. Polymorphism</h3>
    <p>Polymorphism allows the same interface or method to behave differently based on the object type.</p>
    <p><strong>Example:</strong> A <code>calculateFare()</code> method behaving differently for <code>Car</code> and <code>Bike</code>.</p>

    <h3>5. Composition over Inheritance</h3>
    <p>Prefer composition when behavior needs to change dynamically. It leads to more flexible and loosely coupled designs.</p>
    <p><strong>Example:</strong> Injecting a <code>PaymentStrategy</code> instead of extending multiple payment classes.</p>

    <p><strong>Why OOPS matters in LLD:</strong> Interviewers look for how well you model real-world problems, enforce boundaries, and design for change.</p>

    <p><strong>Key takeaway:</strong> Use OOPS concepts intentionally. Clean design beats clever design every time.</p>
  `
},
//  <div class="inpost-img" data-placeholder="https://images.unsplash.com/photo-1518779578993-ec3579fee39f?q=10&w=80&auto=format&fit=crop&crop=entropy">
//       <img data-src="https://images.unsplash.com/photo-1518779578993-ec3579fee39f?q=80&w=1200&auto=format&fit=crop&crop=entropy" alt="OOPS concepts diagram">
//     </div>

  {
  id:'post-1',
  title:'Low Level Design (LLD): Turning Requirements into Clean Code',
  author:'Saravana Kumar',
  date:'2025-09-12',
  tags:['lld','system-design','oop','interview-prep'],
  image:'https://images.unsplash.com/photo-1555066931-4365d14bab8c?q=80&w=1600&auto=format&fit=crop&crop=entropy',
  placeholder:'https://images.unsplash.com/photo-1555066931-4365d14bab8c?q=10&w=40&auto=format&fit=crop&crop=entropy',
  excerpt:'LLD is where design meets implementation. Learn how to identify classes, define responsibilities, and apply SOLID without over-engineering.',
  content:`
    <p>Low Level Design (LLD) is the phase where abstract requirements are translated into concrete classes, methods, and relationships. In interviews, LLD tests how well you can structure code that is readable, extensible, and maintainable.</p>
    <h3>1. Identify Core Entities</h3>
    <p>Start by extracting nouns from the problem statement. These often map to classes. For example, in a Parking Lot system: <strong>ParkingLot</strong>, <strong>Vehicle</strong>, <strong>Slot</strong>, <strong>Ticket</strong>.</p>

    <h3>2. Define Responsibilities</h3>
    <p>Each class should have a single reason to change. Avoid god classes. Ask: <em>“What is this class responsible for?”</em></p>

    <h3>3. Apply SOLID Principles (Pragmatically)</h3>
    <p>SOLID principles guide good design, but they are not rules to blindly follow. Use them where they reduce coupling and improve clarity. For example, apply <strong>Open-Closed Principle</strong> when new features are expected.</p>

    <h3>4. Define Relationships</h3>
    <p>Use composition over inheritance when behavior varies. Clearly define associations, aggregations, and dependencies between classes.</p>

    <h3>5. Avoid Over-Engineering</h3>
    <p>Do not introduce patterns or abstractions without a clear need. If a design feels hard to explain, it’s probably too complex.</p>

    <h3>6. Design for Extension</h3>
    <p>Your design should allow new requirements with minimal changes. In interviews, explain <em>where</em> changes will go and <em>why</em>.</p>

    <p><strong>Key takeaway:</strong> LLD is not about drawing perfect UML diagrams—it’s about demonstrating clear thinking, responsibility-driven design, and writing code that can evolve.</p>
  `
}


// <div class="inpost-img" data-placeholder="https://images.unsplash.com/photo-1519389950473-47ba0277781c?q=10&w=80&auto=format&fit=crop&crop=entropy">
    //   <img data-src="https://images.unsplash.com/photo-1519389950473-47ba0277781c?q=80&w=1200&auto=format&fit=crop&crop=entropy" alt="LLD class design diagram">
    // </div>

//     id:'post-1', title:'Designing for Focus: How to Build Reader-Friendly UIs', author:'Saravana Kumar', date:'2025-09-09',
//     tags:['design','ux','readability'],
//     image:'https://images.unsplash.com/photo-1496307042754-b4aa456c4a2d?q=80&w=1600&auto=format&fit=crop&crop=entropy',
//     placeholder:'https://images.unsplash.com/photo-1496307042754-b4aa456c4a2d?q=10&w=40&auto=format&fit=crop&crop=entropy',
//     excerpt:'Readers crave focus. Typography, layout and micro-interactions matter.',
//     content:`<p>Readers crave focus. To design a reader-friendly UI, start with typography: use a comfortable measure (line length), slightly larger font sizes for body copy, and increased line-height.</p>
//       <div class="inpost-img" data-placeholder="https://images.unsplash.com/photo-1496307042754-b4aa456c4a2d?q=10&w=80&auto=format&fit=crop&crop=entropy">
//         <img data-src="https://images.unsplash.com/photo-1496307042754-b4aa456c4a2d?q=80&w=1200&auto=format&fit=crop&crop=entropy" alt="typography example">
//       </div>
//       <p>Next, structure content: clear headings, short paragraphs, and callouts help scanability.</p>`
//   }
//   {
//     id:'post-2', title:'Building a Local Dev RAG Stack with Chroma and Ollama', author:'Saravana Kumar', date:'2025-08-30',
//     tags:['llm','rag','devops'],
//     image:'https://images.unsplash.com/photo-1548095115-45697e3c1f6a?q=80&w=1600&auto=format&fit=crop&crop=entropy',
//     placeholder:'https://images.unsplash.com/photo-1548095115-45697e3c1f6a?q=10&w=40&auto=format&fit=crop&crop=entropy',
//     excerpt:'A practical local RAG stack guide.',
//     content:`<p>Key parts: document preprocessing, chunking, vector storage, and your LLM.</p>
//       <div class="inpost-img" data-placeholder="https://images.unsplash.com/photo-1548095115-45697e3c1f6a?q=10&w=80&auto=format&fit=crop&crop=entropy">
//         <img data-src="https://images.unsplash.com/photo-1548095115-45697e3c1f6a?q=80&w=1200&auto=format&fit=crop&crop=entropy" alt="local rag diagram">
//       </div>
//       <p>Keep a clean pipeline and cache embeddings to avoid repeated compute.</p>`
//   },
//   {
//     id:'post-3', title:'Thumbnail Selection Techniques to Improve Click-Through', author:'Priya Menon', date:'2025-07-21',
//     tags:['ml','video','engagement'],
//     image:'https://images.unsplash.com/photo-1524678606370-a47ad25cb82a?q=80&w=1600&auto=format&fit=crop&crop=entropy',
//     placeholder:'https://images.unsplash.com/photo-1524678606370-a47ad25cb82a?q=10&w=40&auto=format&fit=crop&crop=entropy',
//     excerpt:'Choosing thumbnails is both art and science.',
//     content:`<p>Thumbnails are often the first impression. Common heuristics: faces, high-contrast frames, and action frames perform well.</p>`
//   },
//   {
//     id:'post-4',
//     title:'Practical Monitoring for Production ML Systems',
//     author:'Anita Rao',
//     date:'2025-06-10',
//     tags:['ml','monitoring','mle'],
//     image:'https://images.unsplash.com/photo-1557800636-894a64c1696f?q=80&w=1600&auto=format&fit=crop&crop=entropy',
//     placeholder:'https://images.unsplash.com/photo-1557800636-894a64c1696f?q=10&w=40&auto=format&fit=crop&crop=entropy',
//     excerpt:'What to monitor in ML systems and how to set up reliable alerts.',
//     content: `
//       <h2>Monitoring model & data</h2>
//       <p>Monitor both model and data: input distribution drifts, prediction distribution shifts, and downstream KPI changes.</p>

//       <h3>Practical checklist</h3>
//       <ul>
//         <li>Compare input feature distributions vs training set</li>
//         <li>Monitor prediction-confidence and class-frequency changes</li>
//         <li>Track downstream business KPIs and latency</li>
//       </ul>

//       <div class="video-wrapper" aria-hidden="false">
//         <!-- Replace VIDEO_ID with a real YouTube video id -->
//         <iframe src="https://www.youtube.com/embed/pNZw6QxRqfo" title="Monitoring ML systems" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
//       </div>

//       <h3>Alerting</h3>
//       <p>Use anomaly detection + thresholding. Integrate alerts into your on-call workflow (PagerDuty, Slack).</p>

//       <h3>Short case study</h3>
//       <p>Seasonal shifts can cause sudden degradation — add windowed comparison checks and cached-rate evaluation to detect changes quickly.</p>
//     `
//   }
];
