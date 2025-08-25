import Ionicons from "@expo/vector-icons/Ionicons";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Checkbox } from "expo-checkbox";
import { useRouter } from "expo-router";
import { createUserWithEmailAndPassword, getAuth } from "firebase/auth";
import { doc, setDoc } from "firebase/firestore";
import { Formik } from "formik";
import { useRef, useState } from "react";
import {
  Alert,
  Image,
  KeyboardAvoidingView,
  Modal,
  Platform,
  ScrollView,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import logo from "../../assets/images/NexusLogo.png";
import { db } from "../../config/firebaseConfig.js";
import authSchema from "../../utils/authSchema";

const Signup = () => {
  const router = useRouter();
  const auth = getAuth();
  const [isTermsModalVisible, setTermsModalVisible] = useState(false);
  const [hasAgreedToTerms, setHasAgreedToTerms] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const numberInputRef = useRef(null);
  const emailInputRef = useRef(null);
  const passwordInputRef = useRef(null);

  const handleSignup = async (values) => {
    try {
      const userCredentials = await createUserWithEmailAndPassword(
        auth,
        values.email,
        values.password
      );
      const user = userCredentials.user;
      await setDoc(doc(db, "users", user.uid), {
        name: values.name,
        number: values.number,
        email: values.email,
        createdAt: new Date(),
      });
      await AsyncStorage.setItem("userEmail", values.email);
      console.log("Signed In successfully");
      router.push("/home");
    } catch (error) {
      console.log(error);
      if (error.code === "auth/email-already-in-use") {
        Alert.alert("SignUp Failed", "Email already in use", [{ text: "OK" }]);
      } else {
        Alert.alert(
          "SignUp Error",
          "An unexpected error occured Please try again",
          [{ text: "OK" }]
        );
      }
    }
  };

  const handleAgreeToTerms = () => {
    setHasAgreedToTerms(true);
    setTermsModalVisible(false);
  };

  return (
    <SafeAreaView className="bg-[#030303] flex-1">
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={{ flex: 1 }}
      >
        <ScrollView
          contentContainerStyle={{ flexGrow: 1 }}
          keyboardShouldPersistTaps="handled"
        >
          <View className="flex justify-center items-center px-6">
            <View className="flex flex-col items-center mb-7">
              <TouchableOpacity onPress={() => router.push("/")}>
                <Image source={logo} style={{ height: 120, width: 120 }} />
              </TouchableOpacity>
              <Text className="text-[#F1EFEC] text-5xl font-extrabold tracking-widest">
                Nexus
              </Text>
              <Text className="text-[#D4C9BE] text-base mt-2">
                Your finances, connected.
              </Text>
            </View>

            <Text className="text-[#F1EFEC] text-3xl font-semibold mb-6">
              Sign Up
            </Text>

            <View className="w-full max-w-md">
              <Formik
                initialValues={{
                  name: "",
                  number: "",
                  email: "",
                  password: "",
                }}
                onSubmit={handleSignup}
                validationSchema={authSchema}
              >
                {({
                  handleChange,
                  handleBlur,
                  values,
                  errors,
                  touched,
                  handleSubmit,
                }) => (
                  <View className="space-y-5">
                    <View>
                      <Text className="text-[#F1EFEC] mb-2 font-medium">
                        Full Name
                      </Text>
                      <TextInput
                        className="border border-[#D4C9BE] h-12 px-4 rounded-xl text-[#F1EFEC] bg-[#1a1a1a]"
                        onChangeText={handleChange("name")}
                        onBlur={handleBlur("name")}
                        value={values.name}
                        returnKeyType="next"
                        onSubmitEditing={() => {
                          numberInputRef.current?.focus();
                        }}
                      />
                      {touched.name && errors.name && (
                        <Text className="text-red-500 text-xs mt-1">
                          {errors.name}
                        </Text>
                      )}
                    </View>
                    <View>
                      <Text className="text-[#F1EFEC] my-2 font-medium">
                        Phone Number
                      </Text>
                      <TextInput
                        ref={numberInputRef}
                        className="border border-[#D4C9BE] h-12 px-4 rounded-xl text-[#F1EFEC] bg-[#1a1a1a]"
                        onChangeText={handleChange("number")}
                        onBlur={handleBlur("number")}
                        value={values.number}
                        keyboardType="phone-pad"
                        returnKeyType="next"
                        onSubmitEditing={() => {
                          emailInputRef.current?.focus();
                        }}
                      />
                      {touched.number && errors.number && (
                        <Text className="text-red-500 text-xs mt-1">
                          {errors.number}
                        </Text>
                      )}
                    </View>
                    <View>
                      <Text className="text-[#F1EFEC] my-2 font-medium">
                        Email
                      </Text>
                      <TextInput
                        ref={emailInputRef}
                        className="border border-[#D4C9BE] h-12 px-4 rounded-xl text-[#F1EFEC] bg-[#1a1a1a]"
                        onChangeText={handleChange("email")}
                        onBlur={handleBlur("email")}
                        value={values.email}
                        keyboardType="email-address"
                        returnKeyType="next"
                        onSubmitEditing={() => {
                          passwordInputRef.current?.focus();
                        }}
                      />
                      {touched.email && errors.email && (
                        <Text className="text-red-500 text-xs mt-1">
                          {errors.email}
                        </Text>
                      )}
                    </View>
                    <View>
                      <Text className="text-[#F1EFEC] my-2 font-medium">
                        Password
                      </Text>
                      <View className="flex-row items-center border border-[#D4C9BE] rounded-xl bg-[#1a1a1a]">
                        <TextInput
                          ref={passwordInputRef}
                          className="flex-1 h-12 px-4 text-[#F1EFEC]"
                          onChangeText={handleChange("password")}
                          onBlur={handleBlur("password")}
                          value={values.password}
                          secureTextEntry={!showPassword}
                          returnKeyType="done"
                          onSubmitEditing={handleSubmit}
                        />
                        <TouchableOpacity
                          onPress={() => setShowPassword(!showPassword)}
                          className="px-3"
                        >
                          <Ionicons
                            name={showPassword ? "eye" : "eye-off"}
                            size={22}
                            color="#F1EFEC"
                          />
                        </TouchableOpacity>
                      </View>

                      {touched.password && errors.password && (
                        <Text className="text-red-500 text-xs mt-1">
                          {errors.password}
                        </Text>
                      )}
                    </View>
                    <View className="flex-row items-center space-x-3 mt-4">
                      <Checkbox
                        value={hasAgreedToTerms}
                        onValueChange={() =>
                          setHasAgreedToTerms(!hasAgreedToTerms)
                        }
                        color={hasAgreedToTerms ? "#123458" : undefined}
                      />
                      <Text className="text-[#D4C9BE]"> I agree to the </Text>
                      <TouchableOpacity
                        onPress={() => setTermsModalVisible(true)}
                      >
                        <Text className="text-[#F1EFEC] font-bold">
                          Terms and Conditions
                        </Text>
                      </TouchableOpacity>
                    </View>
                    <TouchableOpacity
                      onPress={handleSubmit}
                      disabled={!hasAgreedToTerms}
                      className={`py-3 rounded-2xl mt-4 shadow-lg shadow-black/50 ${hasAgreedToTerms ? "bg-[#123458]" : "bg-gray-500"}`}
                    >
                      <Text className="text-lg text-[#F1EFEC] font-semibold text-center tracking-wide">
                        Sign Up
                      </Text>
                    </TouchableOpacity>
                  </View>
                )}
              </Formik>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
      <Modal
        animationType="slide"
        transparent={true}
        visible={isTermsModalVisible}
        onRequestClose={() => {
          setTermsModalVisible(!isTermsModalVisible);
        }}
      >
        <View className="flex-1 justify-center items-center bg-black/60">
          <View className="w-11/12 bg-[#1a1a1a] rounded-2xl p-6 shadow-lg border border-gray-600">
            <Text className="text-2xl font-bold text-center text-[#F1EFEC] mb-4">
              Terms and Conditions
            </Text>
            <ScrollView
              style={{ maxHeight: 400 }}
              showsVerticalScrollIndicator={false}
            >
              <Text className="text-base text-[#D4C9BE] leading-relaxed">
                <Text className="font-bold">
                  Terms and Conditions for Nexus
                </Text>
                {"\n\n"}
                Last updated: August 25, 2025
                {"\n\n"}
                Welcome to Nexus! These terms and conditions outline the rules
                and regulations for the use of the Nexus application. By
                accessing this app, we assume you accept these terms and
                conditions. Do not continue to use Nexus if you do not agree to
                all of the terms and conditions stated on this page.
                {"\n\n"}
                <Text className="font-bold">1. Accounts</Text>
                {"\n"}
                When you create an account with us, you must provide information
                that is accurate, complete, and current at all times. Failure to
                do so constitutes a breach of the Terms, which may result in
                immediate termination of your account on our service. You are
                responsible for safeguarding the password that you use to access
                the service and for any activities or actions under your
                password.
                {"\n\n"}
                <Text className="font-bold">2. Intellectual Property</Text>
                {"\n"}
                The service and its original content, features, and
                functionality are and will remain the exclusive property of
                Nexus and its licensors. The service is protected by copyright,
                trademark, and other laws of both India and foreign countries.
                {"\n\n"}
                <Text className="font-bold">3. Termination</Text>
                {"\n"}
                We may terminate or suspend your account immediately, without
                prior notice or liability, for any reason whatsoever, including
                without limitation if you breach the Terms. Upon termination,
                your right to use the service will immediately cease.
                {"\n\n"}
                <Text className="font-bold">4. Limitation of Liability</Text>
                {"\n"}
                In no event shall Nexus, nor its directors, employees, partners,
                agents, suppliers, or affiliates, be liable for any indirect,
                incidental, special, consequential or punitive damages,
                including without limitation, loss of profits, data, use,
                goodwill, or other intangible losses, resulting from your access
                to or use of or inability to access or use the service.
                {"\n\n"}
                <Text className="font-bold">5. Changes to These Terms</Text>
                {"\n"}
                We reserve the right, at our sole discretion, to modify or
                replace these Terms at any time. We will try to provide at least
                30 days, notice prior to any new terms taking effect. What
                constitutes a material change will be determined at our sole
                discretion.
                {"\n\n"}
                <Text className="font-bold">6. Contact Us</Text>
                {"\n"}
                If you have any questions about these Terms, please contact us
                at: support@nexusapp.com
              </Text>
            </ScrollView>
            <TouchableOpacity
              onPress={handleAgreeToTerms}
              className="bg-[#123458] py-3 rounded-2xl mt-6"
            >
              <Text className="text-lg text-[#F1EFEC] font-semibold text-center">
                Agree and Continue
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
};

export default Signup;
