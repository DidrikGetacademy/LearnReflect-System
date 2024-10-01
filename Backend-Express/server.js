const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const path = require('path');
const app = express();
const bcrypt = require('bcrypt');
const { resolve } = require("path");
const { error } = require('console');
require("dotenv").config({ path: "./.env" });

// Cors konfigurasjon

const allowedOrigin = ['http://localhost:3001','http://localhost:8081','http://192.168.10.116:3001'];
 const corsOptions = {
  origin: (origin,callback) => {
    if(allowedOrigin.indexOf(origin) !== -1 || !origin){
      callback(null,true);
    } else {
      callback(new Error('not allowed by cors'));
    }
  },
  methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
  Credential: true, // cookies or credentials
 }



  app.use(cors(corsOptions));
  app.use(express.json());



   //mysql setup
  const db = mysql.createConnection({
    host: process.env.DB_HOST,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    database: process.env.DB_DATABASE,
});




//post forespørsel til mysql for Contact
app.post('supportRequest', async (req,res) => {
  const {email, category, message} = req.body;
  try{
    const sql = "INSERT INTO support_requests (`id`email`category`message)";
    const values = [id,email,category,message]
    await db.promise().query(sql,[values]);
  }catch{
    console.log('Error',db.error)
  }
})


db.connect((err) => {
  if (err) {  
      error('Error connecting to MySQL database:', err);
      return;
  }
  console.log('Connected to MySQL database');
}); 



//post forespørsel til mysql for registrering
 app.post('/Register', async (req, res) => {
  const { name, email, password } = req.body;
  try {
    // sjekker om email allerede eksisterer
    const checkEmailSql = "SELECT * FROM login WHERE email = ?";
    const [result] = await db.promise().query(checkEmailSql, [email]);

    if (result.length > 0) {
      console.log('Registration failed: Email already registered.');
      return res.status(400).json({ message: 'Email already registered' });
    }

    // krypterer passord
    const hashedPassword = await bcrypt.hash(password, 10); 

    // sender brukerdata inn til databasen
    const sql = "INSERT INTO login (`name`, `email`, `password`, `role`) VALUES (?)";
    const values = [name, email, hashedPassword, 'user'];
    await db.promise().query(sql, [values]);

    console.log('User registered:', email);
    return res.status(201).json({ message: 'Registration Successful.' });
  } catch (err) {
    console.error('Database error:', err);
    return res.status(500).json({ message: 'Internal Server error.' });
  } 
});







//post forespørsel til mysql for login
app.post('/login', async (req, res) => {
  const { name, email, password } = req.body;
  try {
    const sql = "SELECT * FROM login WHERE `name` = ? OR `email` = ?";
    const [data] = await db.promise().query(sql, [name, email]);

    if (data.length === 0) {
      return res.json("No user exists");
    }

    const user = data[0];

    //sammenlign det hashedpassordet med brukerpassordet 
    const match = await bcrypt.compare(password, user.password);

    if (match) {
      return res.json({ success: true, userId: user.id, username: user.name });
    } else {
      return res.json("Incorrect password");
    }
  } catch (err) {
    console.error('Database Error:', err);
    return res.json("Error");
  }
});




const stripe = require("stripe")(process.env.STRIPE_SECRET_KEY, {
  apiVersion: "2022-08-01",
});




const staticDir = resolve(__dirname, process.env.STATIC_DIR || '../Frontend/build');
app.use(express.static(staticDir));



app.get("/", (req, res) => {
  const indexPath = resolve(staticDir, "index.html");
  res.sendFile(indexPath);
});



app.get("/config", (req, res) => {
  console.log("Received request to /config");
  res.send({
    publishableKey: process.env.STRIPE_PUBLISHABLE_KEY,
  });
});



  app.post("/create-payment-intent", async (req, res) => {
    try {
      const { amount } = req.body;
      const paymentIntent = await stripe.paymentIntents.create({
        currency: "EUR",
        amount: amount,
        automatic_payment_methods: { enabled: true },
      });

      res.send({
        clientSecret: paymentIntent.client_secret,
      });
    } catch (e) {
      return res.status(400).send({
        error: {
          message: e.message,
        },
      });
    }
  }); 


  const PORT = process.env.PORT || 8081;
  app.listen(PORT, () => {
      console.log("Listening on port", PORT);
  });