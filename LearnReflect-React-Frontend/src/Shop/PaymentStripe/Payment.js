  import React, { useEffect, useState } from "react";
  import { Elements } from "@stripe/react-stripe-js";
  import { loadStripe } from "@stripe/stripe-js";
  import CheckoutForm from '../PaymentStripe/CheckoutForm';
  import { useLocation } from "react-router-dom";
  function Payment() {
    const [stripePromise, setStripePromise] = useState(null);
    const [clientSecret, setClientSecret] = useState("");
    const location = useLocation();
    const { totalAmount } = location.state || { totalAmount: 0};

    
    useEffect(() => {
      fetch("http://localhost:8081/config")
        .then(async (response) => {
          if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
          }
          const { publishableKey } = await response.json();
          setStripePromise(loadStripe(publishableKey));
        })
        .catch((error) => {
          console.error("Error fetching publishable key:", error);
        });
    }, []);




    useEffect(() => {
      fetch("http://localhost:8081/create-payment-intent", {
        method: "POST",
        headers: { 'Content-Type': 'application/json'},
        body: JSON.stringify({ amount: totalAmount * 100}),
      })
        .then(async (response) => {
          if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
          }
          var { clientSecret } = await response.json();
          setClientSecret(clientSecret);
        }).catch((error) => {
          console.error("Error fetching client secret:", error);
        });
    }, [totalAmount]);




    return (
       <>
        {clientSecret && stripePromise && (
          <Elements stripe={stripePromise} options={{ clientSecret }}>
            <CheckoutForm totalAmount={totalAmount} />
          </Elements>
        )}
      </>
    );
    
  }

  export default Payment;
