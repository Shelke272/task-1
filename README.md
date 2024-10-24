<h3>Input :</h3>
from flask import Flask, render_template, request, redirect, url_for, session
from models import db, Product

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key'  # Change this to a random secret key
db.init_app(app)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    products = Product.query.all()
    return render_template('index.html', products=products)

@app.route('/product/<int:product_id>')
def product(product_id):
    product = Product.query.get_or_404(product_id)
    return render_template('product.html', product=product)

@app.route('/cart')
def cart():
    cart_items = session.get('cart', {})
    return render_template('cart.html', cart_items=cart_items)

@app.route('/add_to_cart/<int:product_id>')
def add_to_cart(product_id):
    cart = session.get('cart', {})
    cart[product_id] = cart.get(product_id, 0) + 1
    session['cart'] = cart
    return redirect(url_for('cart'))

@app.route('/remove_from_cart/<int:product_id>')
def remove_from_cart(product_id):
    cart = session.get('cart', {})
    if product_id in cart:

    
        del cart[product_id]
    session['cart'] = cart
    return redirect(url_for('cart'))

if __name__ == '__main__':
    app.run(debug=True)


<h3>Output :</h3>

Welcome to the E-Commerce Store

Product 1
$19.99 [View Details] [Add to Cart]

Product 2
$29.99 [View Details] [Add to Cart]

Product 3
$15.00 [View Details] [Add to Cart]

[View Cart]

