const express = require('express');
const app = express();
const port = 3000;

app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/public/form.html');
});

app.post('/submit', (req, res) => {
  const { name, email } = req.body;
  res.send(`Received: Name: ${name}, Email: ${email}`);
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
