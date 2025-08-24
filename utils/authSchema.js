import * as Yup from "yup";
const authSchema = Yup.object().shape({
  name: Yup.string().required("Name is required"),
  number: Yup.string()
    .required("Phone number is required")
    .matches(/^[0-9]+$/, "Phone number must be numeric")
    .min(10, "Phone number must be at least 10 digits long"),
  email: Yup.string()
    .required("Email is required")
    .email("Invalid email format"),
  password: Yup.string()
    .required("Password is Required")
    .min(6, "Password must be at least 6 characters long"),
});

export default authSchema;
