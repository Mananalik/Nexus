import Ionicons from "@expo/vector-icons/Ionicons";
import { useRef, useState } from "react";
import {
  Alert,
  Image,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";

import AsyncStorage from "@react-native-async-storage/async-storage";
import { useRouter } from "expo-router";
import { getAuth, signInWithEmailAndPassword } from "firebase/auth";
import { doc, getDoc } from "firebase/firestore";
import { Formik } from "formik";
import { SafeAreaView } from "react-native-safe-area-context";
import logo from "../../assets/images/NexusLogo.png";
import { db } from "../../config/firebaseConfig";
import authSchema from "../../utils/authSchema";

const Signin = () => {
  const router = useRouter();
  const auth = getAuth();
  const passwordInputRef = useRef(null);
  const [showPassword, setShowPassword] = useState(false);

  const handleSignin = async (values) => {
    try {
      const userCredentials = await signInWithEmailAndPassword(
        auth,
        values.email,
        values.password
      );
      const user = userCredentials.user;
      const userDoc = await getDoc(doc(db, "users", user.uid));
      if (userDoc.exists()) {
        console.log("User data: ", userDoc.data());
        await AsyncStorage.setItem("userEmail", values.email);
        router.push("/home");
      } else {
        console.log("no such Document");
      }
    } catch (error) {
      console.log(error);
      if (error.code === "auth/invalid-credential") {
        Alert.alert("Sign In Failed", "Incorrect email/password", [
          { text: "OK" },
        ]);
      } else {
        Alert.alert(
          "Sign In Failed",
          "An unexpected error occurred. Please try again.",
          [{ text: "OK" }]
        );
      }
    }
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
          <View className="flex justify-center items-center px-6 py-10">
            <View className="flex items-center mb-16">
              <TouchableOpacity onPress={() => router.push("/")}>
                <Image source={logo} style={{ height: 150, width: 150 }} />
              </TouchableOpacity>
              <Text className="text-[#F1EFEC] text-4xl font-extrabold tracking-widest">
                Nexus
              </Text>
              <Text className="text-[#D4C9BE] text-base mt-2">
                Your finances, connected.
              </Text>
            </View>

            <Text className="text-[#F1EFEC] text-3xl font-semibold mb-8">
              Sign In
            </Text>

            <View className="w-full max-w-md">
              <Formik
                initialValues={{
                  email: "",
                  password: "",
                }}
                onSubmit={handleSignin}
                validationSchema={authSchema.pick(["email", "password"])}
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
                      <Text className="text-[#F1EFEC] my-2 font-medium">
                        Email
                      </Text>
                      <TextInput
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
                    <TouchableOpacity
                      onPress={handleSubmit}
                      className="bg-[#123458] py-3 rounded-2xl mt-4 shadow-lg shadow-black/50"
                    >
                      <Text className="text-lg text-[#F1EFEC] font-semibold text-center tracking-wide">
                        Sign In
                      </Text>
                    </TouchableOpacity>
                  </View>
                )}
              </Formik>
              <View className="flex-row justify-center mt-7">
                <Text className="text-[#D4C9BE] font-medium">
                  Don&#39;t have an account?{" "}
                </Text>
                <TouchableOpacity onPress={() => router.push("/signup")}>
                  <Text className="text-[#F1EFEC] font-bold">Sign Up</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

export default Signin;
