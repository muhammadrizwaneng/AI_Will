const jsdom = require('jsdom');
const { JSDOM } = jsdom;
const fetch = require('node-fetch');

// Send an HTTP GET request to the AliExpress product page
fetch('https://www.aliexpress.com/item/1005004786718754.html?pdp_ext_f=%7B%22ship_from%22:%22CN%22,%22sku_id%22:%2212000030484089554%22%7D&&scm=1007.25281.317569.0&scm_id=1007.25281.317569.0&scm-url=1007.25281.317569.0&pvid=938c5314-8e56-4f05-a5c1-ea982a2c7686&utparam=%257B%2522process_id%2522%253A%2522standard-item-process-1%2522%252C%2522x_object_type%2522%253A%2522product%2522%252C%2522pvid%2522%253A%2522938c5314-8e56-4f05-a5c1-ea982a2c7686%2522%252C%2522belongs%2522%253A%255B%257B%2522id%2522%253A%252232195265%2522%252C%2522type%2522%253A%2522dataset%2522%257D%255D%252C%2522pageSize%2522%253A%252212%2522%252C%2522language%2522%253A%2522en%2522%252C%2522scm%2522%253A%25221007.25281.317569.0%2522%252C%2522countryId%2522%253A%2522PK%2522%252C%2522scene%2522%253A%2522TopSelection-Waterfall%2522%252C%2522tpp_buckets%2522%253A%252221669%25230%2523265320%252378_21669%25234190%252319165%2523794_15281%25230%2523317569%25230%2522%252C%2522x_object_id%2522%253A%25221005004786718754%2522%257D&pdp_npi=3%40dis%21PKR%21PKR%203%2C785%21PKR%20632%21%21%21%21%21%402103226116941737875517501ed4bc%2112000030484089554%21gdf%21%21&spm=a2g0o.tm1000001522.6946203670.d1&aecmd=true')
  .then(response => response.text())
  .then(data => {
    // Parse the HTML content using jsdom
    const dom = new JSDOM(data);
    const document = dom.window.document;

    // Extract the price, description, and title
    const price = document.querySelector('.price--current--H7sGzqb');
    const description = document.querySelector('[data-pl="product-title"]');
    const title = document.querySelector('title');

    // Display the extracted data
    if (price) {
      console.log('Price:', price.textContent.trim());
    } else {
      console.log('Price not found.');
    }

    if (description) {
      console.log('Description:', description.textContent.trim());
    } else {
      console.log('Description not found.');
    }

    if (title) {
      console.log('Title:', title.textContent.trim());
    } else {
      console.log('Title not found.');
    }
  })
  .catch(error => {
    console.error('An error occurred:', error);
  });
