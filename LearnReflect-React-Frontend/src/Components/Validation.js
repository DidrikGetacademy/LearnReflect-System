export const validateEmail = email => {
  const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return regex.test(String(email).toLowerCase());
};

export const ValidatePassword = password => {
  const IncludesUppercase = /[A-Z]/.test(password);
  const IncludesNumber = /[0-9]/.test(password);
  const IsValidLength = password.length >= 8;
  return IncludesUppercase && IncludesNumber && IsValidLength;
};

export const EqualPasswords = (password, ConfirmPassword) => {
  return password === ConfirmPassword;
};

export const EmptyInput = (username,password) => {
  return username === '' || password === '';
}