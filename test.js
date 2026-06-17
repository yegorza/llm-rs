const { generate } = require("./llm-rs.node");

const result = generate("Hello world", 10);
console.log(result);
