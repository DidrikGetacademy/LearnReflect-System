import React from "react";
import { useLocation } from "react-router-dom";

function ProductCard() {
  const location = useLocation();
  const { product } = location.state || {};

  if (!product) {
    return <div>No product selected</div>;
  }
  return (
    <div className="page-container">
      <div className="product-container">
        <div className="Product-cart">
          <h1 className="title-productcard">
            {product.title}
          </h1>
          <img
            className="productcard-img"
            src={product.imageUrl}
            alt={product.title}
          />
          <p className="productcard-size" class="pick">
            Choose Size
          </p>
          <div class="sizes">
            <div class="size">Small</div>
            <div class="size">Medium</div>
            <div class="size">XL</div>
            <div class="size">XXL</div>
            <div />
          </div>
          <div className="priceanddescription-productcard">
            <p className="description-productcard">
              {product.description}
            </p>
            <p className="price-productcard">
              Price: ${product.Price}
            </p>
          </div>
          <button>Add Cart</button>
        </div>
      </div>
    </div>
  );
}

export default ProductCard;
